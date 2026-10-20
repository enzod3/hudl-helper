"""Cross-view association: 0.5 s window matching stitched into entities, time-disjoint merging, single-view pairing.

The implementation lives in reconstruction/recon.py (one pass over a play); this module re-exports it.
"""
import os, sys

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))
from recon import window_associate, merge_time_disjoint, pair_single_view, team_compatible, classify_teams  # noqa: E402,F401
