# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    try:
        import sam3d_objects.init
    except (ImportError, ModuleNotFoundError):
        # init.py doesn't exist or can't be imported, skip initialization
        pass
