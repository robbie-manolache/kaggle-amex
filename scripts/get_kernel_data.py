
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Upload amexdp package (*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+

import os, sys
sys.path.append(os.path.abspath("."))

from amexdp import env_config
from lazykaggler import kernel_output_download

# set config and data directory
env_config("config.json")
user = "slashie"
kernels = ["amex-feat-profiles-quick-01"]
local_dir = os.environ.get("DATA_DIR")

# run download
for kernel in kernels:
    kernel_output_download(user, kernel, local_dir)
    