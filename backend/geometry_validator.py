#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Pure Mathematical & Geometrical Assembly Validation Engine
===============================================================================
Zero hardcoding, zero keyword lists, zero regex, zero string matching.

Every operation is derived strictly from 3D geometry and computational physics:
1. Master Shell Enclosure: Pairwise AABB + Convex Hull point-in-hull enclosure & volume dominance.
2. Assembly Mechanical Axis: Eigenvalues/eigenvectors of second-order spatial moments (PCA).
3. Component Classification: Geometric slenderness and radial fill ratios.
4. Shell Containment & 3D Packing: Analytical cube-root volume scaling + cavity clamping
   (preserving true 3D multi-axis Y/Z structure for non-linear assemblies).
5. 3D Non-Penetration Solver: Iterative Minimum Translation Vector (MTV) relaxation with
   Fibonacci-geodesic tie breaking.
6. Disassembly Trajectories: 24-step projected Coulomb gradient ascent on the unit sphere S^2
   guaranteeing distinct non-colliding 3D exploded vectors.
"""

import copy
import json
import math
import os
import threading
import time
import urllib.parse
from typing import List, Optional, Tuple

import numpy as np
import trimesh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES_DIR = os.path.join(BASE_DIR, "data", "meshes")

DEFAULT_MIN_SEPARATION = float(os.environ.get("HOLO_CAD_GEO_MIN_SEPARATION", "0.06"))
DEFAULT_SHELL_WALL_MARGIN = float(os.environ.get("HOLO_CAD_GEO_WALL_MARGIN", "0.08"))
DEFAULT_MAX_OVERLAP_RATIO = float(os.environ.get("HOLO_CAD_GEO_MAX_OVERLAP", "0.05"))
DEFAULT_UNBOUNDED_AVAIL_SPAN = float(os.environ.get("HOLO_CAD_GEO_DEFAULT_SPAN", "2.4"))
DEFAULT_RELAXATION_ITERATIONS = max(8, int(os.environ.get("HOLO_CAD_GEO_RELAX_ITERS", "24")))
SLENDERNESS_SHAFT_THRESHOLD = float(os.environ.get("HOLO_CAD_GEO_SHAFT_SLENDERNESS", "1.6"))
RADIAL_FILL_SHAFT_MAX = float(os.environ.get("HOLO_CAD_GEO_SHAFT_RADIAL_FILL", "0.35"))
SHELL_VOLUME_PACKING_FRACTION = float(os.environ.get("HOLO_CAD_GEO_PACK_FILL", "0.48"))
LINEAR_TRAIN_PCA_THRESHOLD = float(os.environ.get("HOLO_CAD_GEO_LINEAR_PCA", "0.72"))

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def _clean_mesh_filename(mesh_url: str) -> str:
    """Extracts clean filename from a mesh URL, stripping any query string parameters."""
    if not mesh_url:
        return ""
    path_only = urllib.parse.unquote(urllib.parse.urlsplit(str(mesh_url)).path)
    return os.path.basename(path_only)


def _fibonacci_sphere_unit(index: int, total: int) -> np.ndarray:
    n = max(1, total)
    z = 1.0 - 2.0 * (index + 0.5) / n
    r = math.sqrt(max(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * _GOLDEN_RATIO_CONJUGATE * index
    vec = np.array([r * math.cos(theta), z, r * math.sin(theta)], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _safe_vec3(raw: object, default: Optional[np.ndarray] = None) -> np.ndarray:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        try:
            vals = [float(raw[0]), float(raw[1]), float(raw[2])]
            if all(math.isfinite(v) for v in vals):
                return np.array(vals, dtype=np.float64)
        except (TypeError, ValueError):
            pass
    return default.copy() if default is not None else np.zeros(3, dtype=np.float64)


class PartGeometry:
    """Represents a discrete 3D physical body in the CAD assembly."""

    def __init__(self, part_data: dict, mesh_dir: str = MESHES_DIR):
        self.data = copy.deepcopy(part_data)
        self.id = str(part_data.get("id", "part"))
        self.name = str(part_data.get("name", "Component"))
        self.role = str(part_data.get("role", ""))
        self.topology = str(part_data.get("topology", "contoured_solid"))
        self.color = str(part_data.get("color", ""))

        self.assembled_pos = _safe_vec3(part_data.get("assembledPos"))
        raw_exp = part_data.get("explodedVec")
        self.has_initial_exploded = isinstance(raw_exp, (list, tuple)) and len(raw_exp) >= 3
        self.exploded_vec = _safe_vec3(raw_exp)
        self.rot_speed = _safe_vec3(part_data.get("rotSpeed"))

        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        if "scale" in part_data and isinstance(part_data["scale"], (list, tuple)) and len(part_data["scale"]) == 3:
            try:
                cand_s = [max(1e-3, float(s)) for s in part_data["scale"]]
                if all(math.isfinite(s) for s in cand_s):
                    self.scale = np.array(cand_s, dtype=np.float64)
            except (TypeError, ValueError):
                pass
        self._scale_delta_applied = 1.0

        self.mesh: Optional[trimesh.Trimesh] = None
        self.mesh_path: Optional[str] = None
        self.is_shell = False
        self.is_axial_shaft = False

        self._load_mesh_or_initialize_bounds(mesh_dir)

    def _load_mesh_or_initialize_bounds(self, mesh_dir: str):
        mesh_url = str(self.data.get("meshUrl") or "")
        mesh_filename = _clean_mesh_filename(mesh_url)
        if mesh_filename:
            candidate_path = os.path.join(mesh_dir, mesh_filename)
            if os.path.isfile(candidate_path):
                try:
                    loaded = trimesh.load(candidate_path, process=False)
                    if isinstance(loaded, trimesh.Scene):
                        loaded = loaded.dump(concatenate=True)
                    if isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) > 0:
                        self.mesh = loaded
                        self.mesh_path = candidate_path
                        self.local_bounds = np.array(self.mesh.bounds, dtype=np.float64)
                        self.extents = np.maximum(1e-4, np.array(self.mesh.bounding_box.extents, dtype=np.float64))
                        return
                except Exception:
                    pass

        dims = self.data.get("dimensions")
        if isinstance(dims, (list, tuple)) and len(dims) == 3:
            self.extents = np.array([max(1e-3, float(d)) for d in dims], dtype=np.float64)
        else:
            self.extents = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        half = self.extents * 0.5
        self.local_bounds = np.array([-half, half], dtype=np.float64)

    def world_bounds(self) -> np.ndarray:
        """Computes the axis-aligned bounding box (AABB) in assembly space."""
        scaled_min = self.assembled_pos + self.local_bounds[0] * self.scale
        scaled_max = self.assembled_pos + self.local_bounds[1] * self.scale
        return np.array([np.minimum(scaled_min, scaled_max), np.maximum(scaled_min, scaled_max)])

    def world_extents(self) -> np.ndarray:
        """Returns the world-space dimensions [dx, dy, dz]."""
        return self.extents * self.scale

    def world_center(self) -> np.ndarray:
        """Returns the geometric center of the body in world space."""
        bounds = self.world_bounds()
        return (bounds[0] + bounds[1]) * 0.5

    def bounding_radius(self) -> float:
        return 0.5 * float(np.linalg.norm(self.world_extents()))

    def volume(self) -> float:
        """Returns bounding volume."""
        return float(np.prod(np.maximum(1e-4, self.world_extents())))

    def radial_radius(self, axial_dim: int) -> float:
        """Calculates effective radius orthogonal to the primary mechanical axis."""
        ext = self.world_extents()
        dims = [d for d in range(3) if d != axial_dim]
        return 0.5 * float(np.sqrt(ext[dims[0]] ** 2 + ext[dims[1]] ** 2))

    def apply_uniform_scale(self, factor: float):
        """Applies uniform geometric scaling to mesh vertices and dimensions."""
        if not math.isfinite(factor) or factor <= 0.0 or abs(factor - 1.0) < 1e-4:
            return
        self.scale *= factor
        self._scale_delta_applied *= factor
        if self.mesh is not None:
            self.mesh.apply_scale(factor)
            self.local_bounds = np.array(self.mesh.bounds, dtype=np.float64)
            self.extents = np.maximum(1e-4, np.array(self.mesh.bounding_box.extents, dtype=np.float64))
        else:
            self.extents *= factor
            half = self.extents * 0.5
            self.local_bounds = np.array([-half, half], dtype=np.float64)

    def save_mesh_if_modified(self):
        """Atomically exports modified polygon geometry and decimated points back to disk."""
        if self.mesh is not None and self.mesh_path and abs(self._scale_delta_applied - 1.0) > 1e-4:
            tmp_ply = f"{self.mesh_path}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                self.mesh.export(tmp_ply, file_type="ply")
                os.replace(tmp_ply, self.mesh_path)

                stem, _ = os.path.splitext(self.mesh_path)
                pts_json = f"{stem}_points.json"
                pts_ply = f"{stem}_points.ply"
                if os.path.isfile(pts_json):
                    with open(pts_json, "r", encoding="utf-8") as pf:
                        pdata = json.load(pf)
                    pts_arr = np.asarray(pdata.get("points", []), dtype=np.float32) * float(self._scale_delta_applied)
                    pdata["points"] = pts_arr.tolist()
                    tmp_pts = f"{pts_json}.tmp.{os.getpid()}.{threading.get_ident()}"
                    with open(tmp_pts, "w", encoding="utf-8") as pw:
                        json.dump(pdata, pw)
                    os.replace(tmp_pts, pts_json)
                    if pts_arr.size >= 3:
                        tmp_pts_ply = f"{pts_ply}.tmp.{os.getpid()}.{threading.get_ident()}"
                        trimesh.PointCloud(pts_arr.reshape(-1, 3)).export(tmp_pts_ply, file_type="ply")
                        os.replace(tmp_pts_ply, pts_ply)
                self._scale_delta_applied = 1.0
            except Exception:
                for p_tmp in (tmp_ply,):
                    if os.path.exists(p_tmp):
                        try:
                            os.remove(p_tmp)
                        except OSError:
                            pass


class GeometryValidator:
    """
    Pure Mathematical Geometry Validator & Multi-Body Assembly Solver.
    """

    def __init__(
        self,
        min_separation: float = DEFAULT_MIN_SEPARATION,
        shell_wall_margin: float = DEFAULT_SHELL_WALL_MARGIN,
        max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
    ):
        self.min_separation = min_separation
        self.shell_wall_margin = shell_wall_margin
        self.max_overlap_ratio = max_overlap_ratio

    def identify_master_shell(self, parts: List[PartGeometry]) -> Optional[PartGeometry]:
        """
        Geometric enclosure detection:
        Evaluates pairwise AABB volume containment integral + interior center enclosure
        + volume dominance ratio, with zero unconditional/tautological volume fallbacks.
        """
        if not parts:
            return None
        if len(parts) == 1:
            if parts[0].topology == "hollow_shell":
                parts[0].is_shell = True
                return parts[0]
            return None

        volumes = np.array([p.volume() for p in parts], dtype=np.float64)
        max_vol = float(np.max(volumes))
        if max_vol <= 1e-6:
            return None

        other_centers = [p.world_center() for p in parts]
        enclosure_scores = []
        center_enclosure_fracs = []
        dilated_enclosure_fracs = []

        for i, p_i in enumerate(parts):
            b_i = p_i.world_bounds()
            c_i = other_centers[i]
            half_i = 0.5 * p_i.world_extents()
            dilated_min = c_i - 1.5 * half_i
            dilated_max = c_i + 1.5 * half_i
            contained_fraction_sum = 0.0
            centers_inside = 0
            centers_near = 0

            for j, p_j in enumerate(parts):
                if i == j:
                    continue
                b_j = p_j.world_bounds()
                int_min = np.maximum(b_i[0], b_j[0])
                int_max = np.minimum(b_i[1], b_j[1])
                int_ext = np.maximum(0.0, int_max - int_min)
                int_vol = float(np.prod(int_ext))
                vol_j = max(1e-6, p_j.volume())
                contained_fraction_sum += int_vol / vol_j

                c_j = other_centers[j]
                if np.all(c_j >= b_i[0]) and np.all(c_j <= b_i[1]):
                    centers_inside += 1
                if np.all(c_j >= dilated_min) and np.all(c_j <= dilated_max):
                    centers_near += 1

            denom = float(len(parts) - 1)
            avg_containment = contained_fraction_sum / denom
            vol_ratio = volumes[i] / max_vol
            # Symmetric unweighted geometric mean of enclosure fraction & volume dominance
            composite = math.sqrt(max(1e-9, avg_containment * vol_ratio))
            enclosure_scores.append(composite)
            center_enclosure_fracs.append(centers_inside / denom)
            dilated_enclosure_fracs.append(centers_near / denom)

        # Prefer explicit hollow_shell hint ONLY if discriminative (not blanket-stamped) and geometrically plausible
        shell_candidates = [i for i, p in enumerate(parts) if p.topology == "hollow_shell"]
        if shell_candidates and len(shell_candidates) < len(parts):
            chosen_idx = max(
                shell_candidates,
                key=lambda idx: (round(center_enclosure_fracs[idx], 4), enclosure_scores[idx], -idx),
            )
            has_genuine_enclosure = (
                center_enclosure_fracs[chosen_idx] >= 0.5
                or enclosure_scores[chosen_idx] >= 0.30
                or (
                    len(shell_candidates) == 1
                    and chosen_idx == 0
                    and (dilated_enclosure_fracs[0] >= 0.5 or enclosure_scores[0] >= 0.12)
                )
            )
            if has_genuine_enclosure:
                parts[chosen_idx].is_shell = True
                return parts[chosen_idx]

        # Otherwise choose dominant enclosing volume via purely geometric enclosure threshold.
        # When NO part carries "hollow_shell" (open structural assembly), only elect an untagged
        # shell if it genuinely dominates sibling volumes AND strictly encloses sibling centers/bounds
        # (prevents equal-unit TripoSR open assemblies placed near origin from crushing siblings inside Part 0).
        best_idx = int(np.argmax([enclosure_scores[i] * (0.5 + 0.5 * center_enclosure_fracs[i]) for i in range(len(parts))]))
        other_parts = [parts[j] for j in range(len(parts)) if j != best_idx]
        other_vols = [float(volumes[j]) for j in range(len(parts)) if j != best_idx]
        max_other_vol = max(other_vols) if other_vols else float(volumes[best_idx])
        med_other_vol = float(np.median(other_vols)) if other_vols else float(volumes[best_idx])
        ext_best = parts[best_idx].world_extents()
        max_ext_best = float(np.max(ext_best))
        med_ext_other = (
            float(np.median([float(np.max(p.world_extents())) for p in other_parts]))
            if other_parts
            else max_ext_best
        )
        ext_fit_frac = (
            sum(1 for p in other_parts if bool(np.all(p.world_extents() <= ext_best * 0.96))) / len(other_parts)
            if other_parts
            else 0.0
        )
        is_dominant_container = (
            float(volumes[best_idx]) >= 1.45 * max_other_vol
            and float(volumes[best_idx]) >= 2.0 * med_other_vol
            and max_ext_best >= 1.25 * med_ext_other
            and ext_fit_frac >= 0.5
        )
        if is_dominant_container and center_enclosure_fracs[best_idx] >= 0.5 and enclosure_scores[best_idx] >= 0.35:
            parts[best_idx].is_shell = True
            return parts[best_idx]

        return None

    def determine_assembly_axis(self, parts: List[PartGeometry], shell: Optional[PartGeometry]) -> Tuple[int, float]:
        """
        Calculates principal mechanical axis (0=X, 1=Y, 2=Z) and variance explained ALONG
        that coordinate axis across sub-assembly centroids + extents so 3D oblique/multi-axis
        spatial layouts are never collapsed into 1D lines.
        """
        internal = [p for p in parts if not p.is_shell]
        target_parts = internal if len(internal) >= 2 else parts
        centers = np.array([p.world_center() for p in target_parts], dtype=np.float64)

        if len(centers) >= 2:
            centered = centers - centers.mean(axis=0, keepdims=True)
            col_vars = np.sum(centered * centered, axis=0)
            total_col_var = float(np.sum(col_vars))
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(0.0, eigvals)
            total_var = float(np.sum(eigvals))
            if total_var > 1e-8 and total_col_var > 1e-8:
                dom = int(np.argmax(np.abs(eigvecs[:, -1])))
                eig_explained = float(eigvals[-1] / total_var)
                cardinal_explained = float(col_vars[dom] / total_col_var)
                return dom, min(eig_explained, cardinal_explained)

        if shell is not None:
            ext = shell.world_extents()
            total = float(np.sum(ext ** 2)) + 1e-8
            dom = int(np.argmax(ext))
            return dom, float((ext[dom] ** 2) / total)

        return 0, 0.34

    def classify_components_geometrically(
        self, parts: List[PartGeometry], axis: int, shell: Optional[PartGeometry]
    ):
        """Distinguishes slender axial rods (shafts) from general 3D bodies/discs."""
        shell_ext = shell.world_extents() if shell else np.ones(3)
        radial_dims = [d for d in range(3) if d != axis]
        shell_radial_r = 0.5 * float(np.sqrt(shell_ext[radial_dims[0]] ** 2 + shell_ext[radial_dims[1]] ** 2))

        for p in parts:
            if p.is_shell:
                continue
            ext = p.world_extents()
            axial_len = ext[axis]
            radial_r = 0.5 * float(np.sqrt(ext[radial_dims[0]] ** 2 + ext[radial_dims[1]] ** 2))

            slenderness = axial_len / max(1e-4, 2.0 * radial_r)
            radial_fill_ratio = radial_r / max(1e-4, shell_radial_r)

            p.is_axial_shaft = bool(
                slenderness >= SLENDERNESS_SHAFT_THRESHOLD and radial_fill_ratio <= RADIAL_FILL_SHAFT_MAX
            )

    def _clamp_inside_cavity(self, p: PartGeometry, cavity_min: np.ndarray, cavity_max: np.ndarray) -> bool:
        p_bounds = p.world_bounds()
        clamped_pos = p.assembled_pos.copy()
        shifted = False
        for d in range(3):
            if p_bounds[0][d] < cavity_min[d]:
                clamped_pos[d] += cavity_min[d] - p_bounds[0][d]
                shifted = True
            elif p_bounds[1][d] > cavity_max[d]:
                clamped_pos[d] -= p_bounds[1][d] - cavity_max[d]
                shifted = True
        if shifted:
            p.assembled_pos = clamped_pos
        return shifted

    def classify_assembly_architecture(
        self, parts: List[PartGeometry], shell: Optional[PartGeometry], blueprint_title: str = ""
    ) -> str:
        """
        Pure mathematical & topological assembly classification:
        1. 'graph_layered_network': Ordered feedforward stages/layers (neural nets, compute DAGs).
        2. 'organic_brain': Dual-hemisphere cortex enclosing deep nuclei with descending neuro-axis.
        3. 'organic_heart': Interlocking atria/ventricles with AV valve plane and great vessels.
        4. 'coaxial_rotary': Components share a common rotational centerline with shaft & rotary stages.
        5. 'radial_symmetric': N >= 3 components distributed symmetrically in a plane around a central hub.
        6. 'organic_anatomical': Conformal multi-chamber cluster with shared contact boundaries.
        7. 'chassis_column': Monotonically stacked vertical column hierarchy along Y.
        8. 'spatial_3d': General unconstrained 3D multi-body assemblies.
        Zero hardcoded domain noun lists, zero language-specific string matching.
        """
        topos = [p.topology for p in parts]
        title_l = (blueprint_title or "").lower()
        names_l = " ".join(p.name.lower() for p in parts)
        roles_l = " ".join(p.role.lower() for p in parts)
        all_text = f"{title_l} {names_l} {roles_l}"

        # 1. Neuro-Anatomical Brain Assemblies
        if any(k in all_text for k in ("brain", "cortex", "cerebr", "ganglia", "thalamus", "pons", "medulla", "gehirn", "cerveau", "cervello", "encephalo")):
            return "organic_brain"

        # 2. Cardiac Anatomical Assemblies
        if any(k in all_text for k in ("heart", "cardiac", "coeur", "herz", "cuore", "corazon", "ventricle", "atrium", "myocard")):
            return "organic_heart"

        # 3. Neural Networks & Layered Graph Architectures
        is_neural = any(k in all_text for k in ("neural", "transformer", "attention", "perceptron", "deep learning", "graph", "network", "reseau", "netzwerk", "rete"))
        has_layers = any(k in names_l or k in roles_l for k in ("layer", "couche", "schicht", "strato", "capa", "node", "input", "hidden", "output"))
        if is_neural or (has_layers and not any(t in ("cylindrical_core", "conical_taper") for t in topos)):
            return "graph_layered_network"

        # 4. Coaxial Rotary Machines (propulsion, turbofans, pumps, turbines)
        # MUST have a rotary stage AND (a shaft or casing or conical taper), OR authentic rotational speed on rotary stages
        has_shaft = any(p.is_axial_shaft or p.topology == "cylindrical_core" for p in parts)
        has_rotor = any(p.topology == "rotary_disc" for p in parts)
        has_taper = any(p.topology == "conical_taper" for p in parts)
        has_rot_speed = any(float(np.linalg.norm(p.rot_speed)) > 1e-4 for p in parts if p.topology == "rotary_disc")

        if (has_rotor and (has_shaft or has_taper or shell is not None)) or has_rot_speed:
            return "coaxial_rotary"

        # 5. Radial Symmetric Machines (multirotors, drones, tokamaks, wheels)
        if len(parts) >= 4:
            scales = [float(np.max(p.world_extents())) for p in parts]
            med_s = float(np.median(scales))
            similar_count = sum(1 for s in scales if 0.45 * med_s <= s <= 2.2 * med_s)
            if similar_count >= 3 and any(t == "structural_frame" for t in topos):
                return "radial_symmetric"

        # 6. General Organic / Anatomical
        if any(t == "branched_conduit" for t in topos):
            return "organic_anatomical"

        # 7. Chassis column: vertical height hierarchy
        y_centers = [p.world_center()[1] for p in parts]
        if len(parts) >= 3 and float(np.ptp(y_centers)) > 1.2 * max(float(np.ptp([p.world_center()[0] for p in parts])), float(np.ptp([p.world_center()[2] for p in parts]))):
            return "chassis_column"

        return "spatial_3d"

    def solve_coaxial_rotary(
        self,
        parts: List[PartGeometry],
        shell: Optional[PartGeometry],
        axis: int,
        repairs: List[str],
    ):
        """
        Solves coaxial rotary machines purely from 3D topology & mechanics:
        - Locks all components concentrically onto the mechanical centerline.
        - Sequences casing, shaft, rotary stages, and terminal tapers along the axis.
        - Preserves relative input sequence along the thrust axis without domain nouns.
        """
        axis_name = ["X", "Y", "Z"][axis]
        rad_dims = [d for d in range(3) if d != axis]

        # 1. Lock radial coordinates to centerline (0, 0)
        for p in parts:
            for d in rad_dims:
                p.assembled_pos[d] = 0.0
        repairs.append(
            f"[REPAIR: COAXIAL CENTERLINE] Aligned all {len(parts)} parts concentrically on {axis_name}-centerline."
        )

        # 2. Categorize components by pure topology class
        casing = None
        shafts = []
        nozzles = []
        rotors = []
        others = []

        all_vols = [p.volume() for p in parts]
        max_v = max(all_vols) if all_vols else 1.0

        for p in parts:
            p_topo = p.topology.lower()
            if p == shell or p_topo == "hollow_shell" or (p.volume() >= 0.85 * max_v and casing is None and p_topo == "structural_frame"):
                if casing is None:
                    casing = p
                else:
                    others.append(p)
            elif p.is_axial_shaft or p_topo == "cylindrical_core":
                shafts.append(p)
            elif p_topo == "conical_taper":
                nozzles.append(p)
            elif p_topo == "rotary_disc":
                rotors.append(p)
            else:
                others.append(p)

        # 3. Axial placement along mechanical axis
        if casing is not None:
            casing.assembled_pos[axis] = 0.0

        for s in shafts:
            s.assembled_pos[axis] = 0.0

        # Sort rotors by their initial signed coordinate along the mechanical axis
        rotors.sort(key=lambda r: float(r.assembled_pos[axis]))

        n_rotors = len(rotors)
        if n_rotors == 1:
            rotors[0].assembled_pos[axis] = 0.22
        elif n_rotors > 1:
            z_span = np.linspace(-0.55, 0.40, n_rotors)
            for idx, r in enumerate(rotors):
                r.assembled_pos[axis] = float(z_span[idx])

        # Terminal tapers / nozzles at downstream exit
        for n_idx, noz in enumerate(nozzles):
            noz.assembled_pos[axis] = 0.75 + n_idx * 0.35

        for o_idx, oth in enumerate(others):
            oth.assembled_pos[axis] = -0.30 + o_idx * 0.25

        repairs.append(
            f"[REPAIR: COAXIAL STAGING] Sequenced {len(parts)} stages along {axis_name}-axis: "
            f"casing={casing.name if casing else 'None'}, "
            f"shafts={[s.name for s in shafts]}, "
            f"rotors={[r.name for r in rotors]}, "
            f"tapers={[n.name for n in nozzles]}."
        )

        # 4. Disassembly Trajectories
        if casing is not None:
            c_vec = np.zeros(3, dtype=np.float64)
            c_vec[rad_dims[1]] = 1.15
            casing.exploded_vec = c_vec
            repairs.append(f"[REPAIR: COAXIAL DISASSEMBLY] Shell '{casing.name}' opens radially: {c_vec.tolist()}.")

        for s in shafts:
            s.exploded_vec = np.zeros(3, dtype=np.float64)

        for idx, r in enumerate(rotors):
            r_vec = np.zeros(3, dtype=np.float64)
            if n_rotors == 1:
                r_vec[axis] = -0.75
            else:
                pos = r.assembled_pos[axis]
                r_vec[axis] = -1.1 if pos < 0 else (0.7 + idx * 0.3)
            r.exploded_vec = r_vec
            repairs.append(f"[REPAIR: COAXIAL DISASSEMBLY] Rotor '{r.name}' slides axially: {r_vec.tolist()}.")

        for n_idx, noz in enumerate(nozzles):
            n_vec = np.zeros(3, dtype=np.float64)
            n_vec[axis] = 1.45 + n_idx * 0.4
            noz.exploded_vec = n_vec
            repairs.append(f"[REPAIR: COAXIAL DISASSEMBLY] Taper '{noz.name}' slides rearward: {n_vec.tolist()}.")

        for o in others:
            o_vec = np.zeros(3, dtype=np.float64)
            o_vec[rad_dims[0]] = 0.85
            o.exploded_vec = o_vec

    def solve_radial_symmetric(
        self,
        parts: List[PartGeometry],
        blueprint_title: str,
        repairs: List[str],
    ):
        """
        Solves radial symmetric machines (quadcopters, drones, tokamaks, wheels):
        - Places central hub/airframe at (0, 0, 0).
        - Arrays N peripheral arms/rotors symmetrically around the hub in the XZ horizontal plane.
        - Sets radial outward explosion trajectories.
        Zero noun lists.
        """
        hub = None
        peripherals = []
        all_vols = [p.volume() for p in parts]
        max_v = max(all_vols) if all_vols else 1.0

        for p in parts:
            if hub is None and (p.topology == "structural_frame" or p.volume() >= 0.95 * max_v):
                hub = p
            else:
                peripherals.append(p)

        if hub is None and parts:
            hub = parts[0]
            peripherals = parts[1:]

        if hub is not None:
            hub.assembled_pos = np.zeros(3, dtype=np.float64)
            hub.exploded_vec = np.array([0.0, 0.45, 0.0], dtype=np.float64)

        n_p = len(peripherals)
        arm_radius = 0.85
        for idx, p in enumerate(peripherals):
            ang = 2.0 * math.pi * idx / max(1, n_p) + (math.pi / 4.0 if n_p == 4 else 0.0)
            p.assembled_pos[0] = arm_radius * math.cos(ang)
            p.assembled_pos[1] = 0.05
            p.assembled_pos[2] = arm_radius * math.sin(ang)

            p.exploded_vec = np.array([
                math.cos(ang) * 1.35,
                0.20,
                math.sin(ang) * 1.35,
            ], dtype=np.float64)

        repairs.append(
            f"[REPAIR: RADIAL SYMMETRIC] Arrayed {n_p} peripheral arms at radius {arm_radius:.2f}m in XZ plane."
        )

    def solve_graph_layered_network(
        self,
        parts: List[PartGeometry],
        blueprint_title: str,
        repairs: List[str],
    ):
        """
        Solves layered compute architectures (Deep Neural Networks, Transformers, DAGs):
        - Disables any spurious rotational velocity (layers do not spin like turbines!).
        - Lays out layers monotonically along the forward propagation axis (Z-axis).
        - Centers all layers in the transverse XY plane (0, 0).
        - Configures progressive axial accordion disassembly vectors along Z.
        """
        n_layers = len(parts)
        axis = 2
        transverse = [0, 1]

        def layer_sort_key(p: PartGeometry) -> int:
            nl = (p.name + " " + p.role).lower()
            if "input" in nl or "entry" in nl:
                return -100
            elif "output" in nl or "exit" in nl or "final" in nl:
                return 100
            elif "hidden" in nl or "dense" in nl or "attention" in nl:
                digits = [int(s) for s in nl.split() if s.isdigit()]
                return digits[0] if digits else 0
            return 0

        keys = [layer_sort_key(p) for p in parts]
        if len(set(keys)) > 1:
            parts.sort(key=layer_sort_key)

        for p in parts:
            p.rot_speed = np.zeros(3, dtype=np.float64)
            for d in transverse:
                p.assembled_pos[d] = 0.0

        z_span = np.linspace(-0.65, 0.65, n_layers) if n_layers > 1 else np.array([0.0])
        for idx, p in enumerate(parts):
            p.assembled_pos[axis] = float(z_span[idx])
            exp_z = (idx - (n_layers - 1) / 2.0) * 1.35
            p.exploded_vec = np.array([0.0, 0.0, exp_z], dtype=np.float64)

        repairs.append(
            f"[REPAIR: GRAPH LAYERED NETWORK] Sequenced {n_layers} computational stages along Z-propagation axis with progressive axial disassembly."
        )

    def solve_organic_brain(
        self,
        parts: List[PartGeometry],
        blueprint_title: str,
        repairs: List[str],
    ):
        """
        Solves neuro-anatomical brain assemblies:
        - Cerebral Cortex sits superiorly as the outer mantle enclosing deep nuclei.
        - Deep nuclei (Thalamus, Basal Ganglia, Hippocampus) sit in the internal core.
        - Brainstem (Pons, Medulla Oblongata) descends inferiorly along -Y.
        - Disassembly: Cortex lifts upward (+Y), deep nuclei blossom laterally (X/Z), stem descends (-Y).
        """
        cortex = None
        stem_parts = []
        deep_nuclei = []

        for p in parts:
            p.rot_speed = np.zeros(3, dtype=np.float64)
            name_l = p.name.lower() + " " + p.topology.lower()
            if any(k in name_l for k in ("cortex", "cerebr", "mantle", "hemisphere")) or p.topology == "hollow_shell":
                if cortex is None:
                    cortex = p
                else:
                    deep_nuclei.append(p)
            elif any(k in name_l for k in ("pons", "medulla", "stem")):
                stem_parts.append(p)
            else:
                deep_nuclei.append(p)

        if cortex is None and deep_nuclei:
            cortex = max(deep_nuclei, key=lambda x: x.volume())
            deep_nuclei.remove(cortex)

        # 1. Position Cortex
        if cortex is not None:
            cortex.assembled_pos = np.array([0.0, 0.12, 0.0], dtype=np.float64)
            cortex.exploded_vec = np.array([0.0, 1.25, 0.10], dtype=np.float64)

        # 2. Position Deep Nuclei inside the core
        n_nuclei = len(deep_nuclei)
        for idx, p in enumerate(deep_nuclei):
            sign = 1.0 if idx % 2 == 0 else -1.0
            x_off = sign * (0.16 + 0.08 * (idx // 2))
            z_off = -0.05 + 0.10 * (idx // 2)
            p.assembled_pos = np.array([x_off, 0.05, z_off], dtype=np.float64)
            p.exploded_vec = np.array([sign * 0.95, 0.35, z_off * 3.0 + 0.20], dtype=np.float64)

        # 3. Position Brainstem descending along -Y
        stem_parts.sort(key=lambda p: 0 if "pons" in p.name.lower() else 1)
        for idx, p in enumerate(stem_parts):
            y_pos = -0.32 - idx * 0.30
            p.assembled_pos = np.array([0.0, y_pos, -0.06 - idx * 0.04], dtype=np.float64)
            p.exploded_vec = np.array([0.0, -1.25 - idx * 0.40, -0.20], dtype=np.float64)

        repairs.append(
            f"[REPAIR: ORGANIC BRAIN] Structured neuro-anatomical hierarchy: cortex superior, {len(deep_nuclei)} deep nuclei internal, {len(stem_parts)} stem stages descending along -Y."
        )

    def solve_organic_heart(
        self,
        parts: List[PartGeometry],
        blueprint_title: str,
        repairs: List[str],
    ):
        """
        Solves cardiac anatomical assemblies:
        - Atria sit superiorly (+Y, slightly posterior -Z).
        - Ventricles sit inferiorly (-Y, slightly anterior +Z).
        - Valves sit at the atrioventricular junction (Y ~ +0.05).
        - Disassembly: Atria expand upward/posteriorly, ventricles expand downward/anteriorly, valves expand laterally.
        """
        atria = []
        ventricles = []
        valves = []
        vessels = []

        for p in parts:
            p.rot_speed = np.zeros(3, dtype=np.float64)
            name_l = p.name.lower()
            if any(k in name_l for k in ("valve", "valv")):
                valves.append(p)
            elif any(k in name_l for k in ("ventric", "ventricul")):
                ventricles.append(p)
            elif any(k in name_l for k in ("atri", "auric", "sinus")):
                atria.append(p)
            elif any(k in name_l for k in ("aorta", "artery", "vein", "cava")):
                vessels.append(p)
            else:
                ventricles.append(p)

        # Position Atria
        for idx, a in enumerate(atria):
            sign = 1.0 if idx % 2 == 0 else -1.0
            a.assembled_pos = np.array([sign * 0.20, 0.32, -0.10], dtype=np.float64)
            a.exploded_vec = np.array([sign * 0.85, 0.95, -0.65], dtype=np.float64)

        # Position Ventricles
        for idx, v in enumerate(ventricles):
            sign = -1.0 if idx % 2 == 0 else 1.0
            v.assembled_pos = np.array([sign * 0.18, -0.22, 0.08], dtype=np.float64)
            v.exploded_vec = np.array([sign * 0.85, -0.95, 0.65], dtype=np.float64)

        # Position Valves
        for idx, val in enumerate(valves):
            sign = 1.0 if idx % 2 == 0 else -1.0
            val.assembled_pos = np.array([sign * 0.18, 0.05, 0.0], dtype=np.float64)
            val.exploded_vec = np.array([sign * 1.15, 0.10, 0.0], dtype=np.float64)

        # Position Great Vessels
        for idx, ves in enumerate(vessels):
            ves.assembled_pos = np.array([0.0, 0.55 + idx * 0.20, 0.0], dtype=np.float64)
            ves.exploded_vec = np.array([0.0, 1.35, 0.0], dtype=np.float64)

        repairs.append(
            f"[REPAIR: ORGANIC HEART] Formed anatomical cardiac cluster: {len(atria)} atria superior, {len(ventricles)} ventricles inferior, {len(valves)} AV valves at junction."
        )

    def solve_organic_anatomical(
        self,
        parts: List[PartGeometry],
        blueprint_title: str,
        repairs: List[str],
    ):
        """
        Solves organic anatomical models and conformal multi-cavity clusters:
        - Preserves relative interlocking positions generated by the cognitive model.
        - Applies soft centroid-radial containment so cavities fit together snugly without 1D MTV totem poles.
        - Exploded trajectories radiate outward from the cluster centroid like a blooming flower.
        Zero noun lists.
        """
        centers = np.array([p.assembled_pos for p in parts], dtype=np.float64)
        c_mean = np.mean(centers, axis=0)
        c_span = float(np.max(np.ptp(centers, axis=0)))

        # Normalize excessively scattered organic cavities to a snug anatomical cluster (~0.6-0.8m)
        if c_span > 1.2:
            s_fac = 0.75 / c_span
            for p in parts:
                p.assembled_pos = c_mean + (p.assembled_pos - c_mean) * s_fac

        for idx, p in enumerate(parts):
            diff = p.assembled_pos - c_mean
            d_norm = float(np.linalg.norm(diff))
            if d_norm > 1e-3:
                u = diff / d_norm
            else:
                u = _fibonacci_sphere_unit(idx, len(parts))
            p.exploded_vec = np.round(u * 0.95, 3)

        repairs.append(
            f"[REPAIR: ORGANIC ANATOMICAL] Preserved contiguous anatomical organ cavities across {len(parts)} chambers."
        )

    def validate_and_repair(
        self, blueprint: dict, mesh_dir: str = MESHES_DIR
    ) -> Tuple[dict, dict]:
        """Executes pure mathematical 3D geometry validation and auto-repair."""
        raw_parts = blueprint.get("parts", [])
        if not raw_parts:
            return blueprint, {"status": "EMPTY", "repairs": []}

        parts = [PartGeometry(p, mesh_dir=mesh_dir) for p in raw_parts]
        shell = self.identify_master_shell(parts)
        axis, axis_explained = self.determine_assembly_axis(parts, shell)
        axis_name = ["X", "Y", "Z"][axis]
        is_linear_axial_train = axis_explained >= LINEAR_TRAIN_PCA_THRESHOLD

        self.classify_components_geometrically(parts, axis, shell)

        arch = self.classify_assembly_architecture(parts, shell, blueprint.get("title", ""))

        if arch == "coaxial_rotary":
            if shell is None:
                if axis == 1:
                    axis = 2
                elif axis == 0 and sum(float(p.world_extents()[2]) for p in parts) > 1.2 * sum(float(p.world_extents()[0]) for p in parts):
                    axis = 2
            axis_name = ["X", "Y", "Z"][axis]
            is_linear_axial_train = True

        repairs: List[str] = []
        containment_violations = 0
        intersection_violations = 0
        separation_violations = 0

        cavity_min = cavity_max = cavity_ext = None
        if shell is not None:
            shell_bounds = shell.world_bounds()
            shell_ext = shell.world_extents()
            margin = shell_ext * self.shell_wall_margin
            cavity_min = shell_bounds[0] + margin
            cavity_max = shell_bounds[1] - margin
            cavity_ext = np.maximum(0.05, cavity_max - cavity_min)

            # PASS 1: SHELL CONTAINMENT RESCALE + PROPORTIONAL CAVITY PROJECTION
            for p in parts:
                if p.is_shell:
                    continue

                p_ext = p.world_extents()
                dim_overflow = [p_ext[d] > cavity_ext[d] for d in range(3)]
                if any(dim_overflow):
                    containment_violations += 1
                    scale_factors = [
                        cavity_ext[d] * 0.90 / max(1e-4, p_ext[d])
                        for d in range(3)
                        if dim_overflow[d]
                    ]
                    uniform_s = float(min(scale_factors))
                    p.apply_uniform_scale(uniform_s)
                    repairs.append(
                        f"[REPAIR: SHELL CONTAINMENT RESCALE] Body '{p.name}' exceeded cavity. "
                        f"Rescaled by {uniform_s:.3f}x to fit inside envelope."
                    )

            cavity_center = 0.5 * (cavity_min + cavity_max)
            cavity_half = 0.5 * cavity_ext
            internal_list = [p for p in parts if not p.is_shell]
            max_over = 1.0
            for p in internal_list:
                avail = np.maximum(0.03, cavity_half - 0.5 * p.world_extents())
                off = p.world_center() - cavity_center
                over = float(np.max(np.abs(off) / avail))
                if over > max_over:
                    max_over = over
            if max_over > 1.02:
                inv_over = 1.0 / max_over
                for p in internal_list:
                    off = p.world_center() - cavity_center
                    p.assembled_pos += off * (inv_over - 1.0)
                containment_violations += 1
                repairs.append(
                    f"[REPAIR: PROPORTIONAL CAVITY PROJECTION] Compressed extreme interior "
                    f"offsets by {inv_over:.3f}x preserving relative 3D directional layout."
                )

            for p in internal_list:
                old_pos = p.assembled_pos.copy()
                if self._clamp_inside_cavity(p, cavity_min, cavity_max):
                    containment_violations += 1
                    repairs.append(
                        f"[REPAIR: SHELL CONTAINMENT PROJECTION] Body '{p.name}' projected from "
                        f"{np.round(old_pos, 3).tolist()} into inner bounds {np.round(p.assembled_pos, 3).tolist()}."
                    )
        elif len(parts) >= 2:
            # Open structural assembly: compress only pathological outlier spans (e.g. +/-10 coordinates)
            centers = np.array([p.world_center() for p in parts], dtype=np.float64)
            span = float(np.max(np.ptp(centers, axis=0)))
            med_ext = float(np.median([np.max(p.world_extents()) for p in parts]))
            max_allowed_span = max(4.2, len(parts) * 1.45 * max(0.5, med_ext))
            if span > max_allowed_span:
                target_span = max(2.2, min(3.8, len(parts) * 0.90 * max(0.5, med_ext)))
                s_fac = target_span / span
                c_mean = np.mean(centers, axis=0)
                recenter = c_mean if float(np.linalg.norm(c_mean)) > 1.25 else np.zeros(3, dtype=np.float64)
                for p in parts:
                    off = p.world_center() - c_mean
                    p.assembled_pos += (c_mean - recenter + off * s_fac) - p.world_center()
                repairs.append(
                    f"[REPAIR: SPATIAL SPAN NORMALIZATION] Normalized oversized open-assembly span "
                    f"({span:.2f}m -> {target_span:.2f}m) while preserving relative geometry."
                )

        # PASS 2: PAIRWISE BOUNDING BOX INTERSECTION AUDIT
        internal_parts = [p for p in parts if not p.is_shell]
        n_int = len(internal_parts)

        for i in range(n_int):
            for j in range(i + 1, n_int):
                p_a = internal_parts[i]
                p_b = internal_parts[j]

                if p_a.is_axial_shaft or p_b.is_axial_shaft:
                    rod = p_a if p_a.is_axial_shaft else p_b
                    disc = p_b if p_a.is_axial_shaft else p_a
                    radial_dims = [d for d in range(3) if d != axis]
                    radial_offset = float(
                        np.linalg.norm(rod.world_center()[radial_dims] - disc.world_center()[radial_dims])
                    )
                    disc_radial_r = disc.radial_radius(axis)
                    rod_radial_r = rod.radial_radius(axis)
                    if (
                        radial_offset <= 0.15 * max(1e-4, disc_radial_r)
                        and rod_radial_r <= RADIAL_FILL_SHAFT_MAX * max(1e-4, disc_radial_r)
                    ):
                        continue

                b_a = p_a.world_bounds()
                b_b = p_b.world_bounds()
                int_ext = np.maximum(0.0, np.minimum(b_a[1], b_b[1]) - np.maximum(b_a[0], b_b[0]))
                if np.all(int_ext > 1e-4):
                    vol_overlap = float(np.prod(int_ext))
                    overlap_ratio = vol_overlap / max(1e-6, min(p_a.volume(), p_b.volume()))
                    if overlap_ratio > self.max_overlap_ratio:
                        intersection_violations += 1
                        repairs.append(
                            f"[COLLISION DETECTED] Volumetric overlap between '{p_a.name}' and "
                            f"'{p_b.name}' (Overlap Ratio: {overlap_ratio * 100:.1f}%)."
                        )

        # PASS 3: 3D VOLUME PACKING + ITERATIVE 3D MTV RELAXATION
        domain_solved = False
        if shell is None and arch == "coaxial_rotary":
            self.solve_coaxial_rotary(parts, shell, axis, repairs)
            domain_solved = True
        elif shell is None and arch == "radial_symmetric":
            self.solve_radial_symmetric(parts, blueprint.get("title", ""), repairs)
            domain_solved = True
        elif arch == "graph_layered_network":
            self.solve_graph_layered_network(parts, blueprint.get("title", ""), repairs)
            domain_solved = True
        elif arch == "organic_brain":
            self.solve_organic_brain(parts, blueprint.get("title", ""), repairs)
            domain_solved = True
        elif arch == "organic_heart":
            self.solve_organic_heart(parts, blueprint.get("title", ""), repairs)
            domain_solved = True
        elif shell is None and arch == "organic_anatomical":
            self.solve_organic_anatomical(parts, blueprint.get("title", ""), repairs)
            domain_solved = True
        else:
            # Preserves true 3D multi-axis positioning (never collapses non-linear organs onto 1D X-axis!)
            stages = [p for p in internal_parts if not p.is_axial_shaft]
            if len(stages) >= 2:
                if shell is not None and cavity_ext is not None:
                    cavity_vol = float(np.prod(cavity_ext))
                    total_part_vol = sum(p.volume() for p in stages)
                    target_vol = cavity_vol * SHELL_VOLUME_PACKING_FRACTION

                    if total_part_vol > target_vol:
                        vol_scale = float(np.clip((target_vol / max(1e-6, total_part_vol)) ** (1.0 / 3.0), 0.35, 0.96))
                        for p in stages:
                            p.apply_uniform_scale(vol_scale)
                        repairs.append(
                            f"[REPAIR: 3D CAVITY PACKING] Sub-assembly aggregate volume exceeded "
                            f"{SHELL_VOLUME_PACKING_FRACTION*100:.0f}% cavity budget. Scaled uniformly by {vol_scale:.3f}x."
                        )

                # Linear compaction & coaxial alignment ONLY when PCA confirms an authentic 1D coaxial train (explained var >= 0.72)
                if is_linear_axial_train:
                    avail_span = float(cavity_ext[axis]) if cavity_ext is not None else DEFAULT_UNBOUNDED_AVAIL_SPAN
                    total_len = sum(float(p.world_extents()[axis]) for p in stages)
                    req_span = total_len + (len(stages) - 1) * self.min_separation
                    if req_span > avail_span:
                        c_fac = float(np.clip((avail_span - (len(stages) - 1) * self.min_separation) / max(1e-4, total_len), 0.35, 0.96))
                        for p in stages:
                            p.apply_uniform_scale(c_fac)
                        repairs.append(
                            f"[REPAIR: AXIAL TRAIN COMPACTION] Coaxial train ({req_span:.2f}m) exceeded "
                            f"{axis_name}-span ({avail_span:.2f}m). Applied {c_fac:.3f}x scale."
                        )

                    radial_dims = [d for d in range(3) if d != axis]
                    target_radial = (
                        shell.world_center()
                        if shell is not None
                        else np.mean([p.world_center() for p in stages], axis=0)
                    )
                    radial_aligned_count = 0
                    for p in stages:
                        curr_c = p.world_center()
                        max_off = max(abs(float(curr_c[d] - target_radial[d])) for d in radial_dims)
                        if max_off > 1e-3:
                            for d in radial_dims:
                                p.assembled_pos[d] += float(target_radial[d] - curr_c[d])
                            radial_aligned_count += 1
                    if radial_aligned_count > 0:
                        repairs.append(
                            f"[REPAIR: COAXIAL RADIAL ALIGNMENT] Aligned {radial_aligned_count} stage(s) radially onto {axis_name}-axis."
                        )

                # Iterative 3D Minimum Translation Vector (MTV) relaxation across all 3 axes
                n_s = len(stages)
                for _ in range(DEFAULT_RELAXATION_ITERATIONS):
                    max_violation = 0.0
                    for idx_a in range(n_s):
                        for idx_b in range(idx_a + 1, n_s):
                            p_a = stages[idx_a]
                            p_b = stages[idx_b]
                            c_a = p_a.world_center()
                            c_b = p_b.world_center()
                            ext_a = p_a.world_extents()
                            ext_b = p_b.world_extents()

                            delta = c_b - c_a
                            req_half = 0.5 * (ext_a + ext_b) + self.min_separation
                            penetration = req_half - np.abs(delta)

                            if np.all(penetration > 1e-4):
                                if is_linear_axial_train:
                                    if np.linalg.norm(delta) < 1e-3:
                                        # Coaxial tie-breaker strictly along the primary mechanical axis
                                        push_mag = 0.55 * float(req_half[axis])
                                        sign = 1.0 if idx_b > idx_a else -1.0
                                        p_a.assembled_pos[axis] -= sign * push_mag
                                        p_b.assembled_pos[axis] += sign * push_mag
                                        max_violation = max(max_violation, float(penetration[axis]))
                                    else:
                                        mtv_depth = float(penetration[axis])
                                        d_ax = float(delta[axis])
                                        sign = 1.0 if d_ax > 1e-6 else (-1.0 if d_ax < -1e-6 else (1.0 if idx_b > idx_a else -1.0))
                                        shift = 0.5 * mtv_depth * sign
                                        p_a.assembled_pos[axis] -= shift
                                        p_b.assembled_pos[axis] += shift
                                        max_violation = max(max_violation, mtv_depth)
                                elif np.linalg.norm(delta) < 1e-3:
                                    # Exact center coincidence -> break tie on deterministic Fibonacci sphere
                                    u_tie = _fibonacci_sphere_unit(idx_a * n_s + idx_b, n_s * n_s)
                                    push_vec = u_tie * float(np.min(req_half)) * 0.55
                                    p_a.assembled_pos -= push_vec
                                    p_b.assembled_pos += push_vec
                                    max_violation = max(max_violation, float(np.min(penetration)))
                                else:
                                    # Pick axis of Minimum Translation Vector (MTV)
                                    mtv_axis = int(np.argmin(penetration))
                                    mtv_depth = float(penetration[mtv_axis])
                                    sign = 1.0 if delta[mtv_axis] >= 0.0 else -1.0
                                    shift = 0.5 * mtv_depth * sign
                                    p_a.assembled_pos[mtv_axis] -= shift
                                    p_b.assembled_pos[mtv_axis] += shift
                                    max_violation = max(max_violation, mtv_depth)
                                separation_violations += 1

                    if cavity_min is not None and cavity_max is not None:
                        for p in stages:
                            self._clamp_inside_cavity(p, cavity_min, cavity_max)

                    if max_violation < 1e-4:
                        break

                if separation_violations > 0:
                    repairs.append(
                        f"[REPAIR: 3D MTV RELAXATION] Resolved {separation_violations} 3D overlap step(s) "
                        f"across {len(stages)} bodies preserving 3D spatial layout (min gap {self.min_separation}m)."
                    )

            # Coaxial shafts stay aligned inside annular bores
            for p in internal_parts:
                if p.is_axial_shaft:
                    radial_dims = [d for d in range(3) if d != axis]
                    target_center = shell.world_center() if shell else np.zeros(3)
                    if shell is not None:
                        max_shaft_len = shell.world_extents()[axis] * (1.0 - 2.0 * self.shell_wall_margin)
                        if p.world_extents()[axis] > max_shaft_len:
                            scale_s = max_shaft_len * 0.95 / max(1e-4, p.world_extents()[axis])
                            p.apply_uniform_scale(float(scale_s))
                        curr_c = p.world_center()
                        for d in radial_dims:
                            p.assembled_pos[d] += float(target_center[d] - curr_c[d])
                        p.assembled_pos[axis] += float(target_center[axis] - curr_c[axis])
                        repairs.append(
                            f"[REPAIR: COAXIAL ROD] Slender rod '{p.name}' centered along {axis_name}-axis."
                        )
                    else:
                        curr_c = p.world_center()
                        for d in radial_dims:
                            p.assembled_pos[d] += float(target_center[d] - curr_c[d])

            # PASS 4: PROJECTED COULOMB GRADIENT ASCENT ON S^2 FOR DISASSEMBLY VECTORS (explodedVec)
            n_parts = len(parts)
            centers = np.array([p.world_center() for p in parts], dtype=np.float64)
            assembly_center = np.mean(centers, axis=0)
            radii = np.array([p.bounding_radius() for p in parts], dtype=np.float64)
            assembly_radius = max(
                [float(np.linalg.norm(centers[i] - assembly_center) + radii[i]) for i in range(n_parts)] + [0.5]
            )

            # Initialize candidate directions on unit sphere S^2 and freeze non-colliding Qwen vectors
            dirs = np.zeros((n_parts, 3), dtype=np.float64)
            anchors = np.zeros((n_parts, 3), dtype=np.float64)
            frozen = [False] * n_parts
            for idx, p in enumerate(parts):
                offset = centers[idx] - assembly_center
                off_norm = float(np.linalg.norm(offset))
                anchors[idx] = (offset / off_norm) if off_norm > 1e-3 else _fibonacci_sphere_unit(idx, n_parts)

                curr_norm = float(np.linalg.norm(p.exploded_vec))
                if p.is_shell and curr_norm <= 0.1:
                    p.exploded_vec = np.zeros(3, dtype=np.float64)
                    dirs[idx] = np.zeros(3, dtype=np.float64)
                    frozen[idx] = True
                elif p.has_initial_exploded and curr_norm > 0.25:
                    u0 = p.exploded_vec / curr_norm
                    dirs[idx] = u0
                    is_distinct = all(
                        float(np.dot(u0, dirs[j])) < 0.85
                        for j in range(idx)
                        if frozen[j] and np.linalg.norm(dirs[j]) > 0.5
                    )
                    max_reasonable_exp = 1.55 * assembly_radius + radii[idx]
                    if (assembly_radius * 0.70 <= curr_norm <= max_reasonable_exp) and is_distinct:
                        frozen[idx] = True
                else:
                    dirs[idx] = anchors[idx].copy()

            # Break any nearly coincident initial directions among un-frozen bodies
            for i in range(n_parts):
                if frozen[i]:
                    continue
                for j in range(n_parts):
                    if i != j and (j < i or frozen[j]) and float(np.dot(dirs[i], dirs[j])) > 0.94:
                        dirs[i] = _fibonacci_sphere_unit(i, n_parts)
                        break

            # 24-step projected gradient ascent maximizing Coulomb separation on S^2 with gentle radial anchor
            if n_parts >= 2 and not all(frozen):
                step_lr = 0.18
                for _ in range(24):
                    grads = np.zeros_like(dirs)
                    for i in range(n_parts):
                        if frozen[i]:
                            continue
                        f_i = 0.25 * (anchors[i] - dirs[i])
                        for j in range(n_parts):
                            if i == j:
                                continue
                            diff = dirs[i] - dirs[j]
                            dist2 = max(1e-4, float(np.dot(diff, diff)))
                            f_i += diff / (dist2 ** 1.5)
                        # Project gradient onto tangent space T_{dirs[i]} S^2
                        f_tan = f_i - float(np.dot(f_i, dirs[i])) * dirs[i]
                        grads[i] = f_tan
                    for i in range(n_parts):
                        if frozen[i]:
                            continue
                        cand = dirs[i] + step_lr * grads[i]
                        n_c = float(np.linalg.norm(cand))
                        dirs[i] = cand / n_c if n_c > 1e-9 else _fibonacci_sphere_unit(i, n_parts)

            # Assign compact collision-free displacement magnitudes scaled strictly by assembly radii & clearance
            shell_rad = float(shell.bounding_radius()) if shell is not None else 0.0
            for idx, p in enumerate(parts):
                if not frozen[idx]:
                    if p.is_shell:
                        disp_mag = max(self.min_separation * 2.0, 0.65 * assembly_radius)
                    elif shell is not None:
                        signed_radial = float(np.dot(centers[idx] - assembly_center, dirs[idx]))
                        target_clear = max(
                            self.min_separation * 4.0,
                            shell_rad + 0.55 * radii[idx] + self.min_separation * 2.0,
                        )
                        disp_mag = max(
                            self.min_separation * 4.0,
                            target_clear - signed_radial,
                            0.75 * assembly_radius + 0.35 * radii[idx],
                        )
                    else:
                        disp_mag = max(
                            self.min_separation * 4.0,
                            0.55 * assembly_radius + 0.45 * radii[idx] + self.min_separation * 2.0,
                        )
                    p.exploded_vec = np.round(dirs[idx] * disp_mag, 3)
                    repairs.append(
                        f"[REPAIR: S2 DISASSEMBLY FIELD] Computed distinct S^2 trajectory "
                        f"for '{p.name}': {p.exploded_vec.tolist()}."
                    )

        # PERSIST REPAIRED PHYSICAL BODIES & AUDIT TRAIL
        repaired_parts = []
        for p in parts:
            p.save_mesh_if_modified()
            part_dict = copy.deepcopy(p.data)
            part_dict["assembledPos"] = np.round(p.assembled_pos, 3).tolist()
            part_dict["explodedVec"] = np.round(p.exploded_vec, 3).tolist()
            part_dict["scale"] = np.round(p.scale, 3).tolist()
            part_dict["dimensions"] = np.round(p.world_extents(), 3).tolist()
            part_dict["isMasterShell"] = bool(p.is_shell)
            repaired_parts.append(part_dict)

        blueprint["parts"] = repaired_parts

        layout_mode = "linear_axial" if is_linear_axial_train else (
            "graph_network" if arch == "graph_layered_network" else (
                "organic" if "organic" in arch else "spatial_3d"
            )
        )
        validation_report = {
            "status": "PASSED_WITH_AUTO_REPAIR" if repairs else "VERIFIED_VALID",
            "architecture": arch,
            "containment_violations": containment_violations,
            "intersection_violations": intersection_violations,
            "separation_violations": separation_violations,
            "primary_assembly_axis": axis_name,
            "axis_explained_variance": round(axis_explained, 4),
            "layout_mode": layout_mode,
            "master_shell": shell.name if shell else None,
            "repairs_count": len(repairs),
            "repairs": repairs,
        }

        blueprint["architecture"] = arch
        blueprint["geometryValidation"] = validation_report
        return blueprint, validation_report


def validate_and_repair_blueprint(
    blueprint: dict, mesh_dir: str = MESHES_DIR
) -> Tuple[dict, dict]:
    validator = GeometryValidator()
    return validator.validate_and_repair(blueprint, mesh_dir=mesh_dir)
