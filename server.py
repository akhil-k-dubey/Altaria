#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Canonical Entrypoint Shim
==============================================
Delegates directly to backend/qwen_service.py so there is strictly one
single source of truth for HTTP routes, SSE streaming, Qwen2.5-VL inference,
TripoSR synthesis, and geometry validation (eliminates split-brain forks).
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    import qwen_service
    qwen_service.main()
else:
    from qwen_service import *  # noqa: F401,F403
