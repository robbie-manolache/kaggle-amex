# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Upload amexdp package (*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+

import os, sys
sys.path.append(os.path.abspath("."))

import shutil
from lazykaggler import gen_dataset_metafile, upload_dataset

from amexdp import env_config
env_config("config.json")

# local destination directory
dst_dir = os.path.join(os.environ.get("DATA_DIR"), "amexdp")

# check that metadata file exists for Kaggle Datasets API else create
if os.path.exists(os.path.join(dst_dir, "dataset-metadata.json")):
    pass
else:
    gen_dataset_metafile(
        local_path=dst_dir,
        user="slashie",
        title="AMEX Default Prediction Package",
        subtitle="Python package for the AMEX Default Prediction Competition"
    )

# get package file name
file_name = "amexdp-0.1.0-py3-none-any.whl"

# copy source file to destination
shutil.copy2(os.path.join("dist", file_name), 
             os.path.join(dst_dir, file_name))

# upload to Kaggle datasets
upload_dataset(dst_dir, 
               new_version=False, 
               version_notes="initial release")
