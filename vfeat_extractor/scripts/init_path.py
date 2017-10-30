"""Init module path for scripts."""

import os
import sys
import pdb


def add_path(path):
    """Add path to system PYTHONPATH."""
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

#pdb.set_trace()
# Add lib to PYTHONPATH
add_path(os.path.join(this_dir, '..'))
