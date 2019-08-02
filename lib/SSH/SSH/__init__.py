# --------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Written by Mahyar Najibi
# --------------------------------------------------------

import sys
# Add caffe and lib to the paths
if not 'lib/SSH/caffe-ssh/python' in sys.path:
    sys.path.insert(0, 'lib/SSH/caffe-ssh/python')
if not 'lib/SSH/lib' in sys.path:
    sys.path.insert(0, 'lib/SSH/lib')
from utils.get_config import cfg

if not cfg.DEBUG:
    import os
    # Suppress Caffe (it does not affect training, only test and demo)
    os.environ['GLOG_minloglevel']='3'