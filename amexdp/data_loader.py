
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Data Loader (*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def init_metadata():
    
    metadata = {
        "keys": ["customer_ID", "S_2"],
        "col_types": {
            "D": "delinquency",
            "S": "spending",
            "P": "payment",
            "B": "balance",
            "R": "risk"
        },
        "cats": ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 
                 'D_117', 'D_120', 'D_126', 'D_63', 
                 'D_64', 'D_66', 'D_68', 'D_87'],
        "cat_maps": {
            "all": {"D_64": {"-1": "NA"},
                    "D_66": {0: -9},
                    "D_68": {0: -9}},
            "last": {"D_126": {-1: 0, -9: 0}}
        },
        "missing": {
            "test": ["B_37", "B_40", "B_41", "D_86",
                     "S_12", "S_17", "S_26"],
            "train": ["R_7", "R_12", "R_14", "R_20"]
        }
    }
    
    return metadata
    

class DataLoader:

    def __init__(self, 
                 feature_dir: str,
                 profile_path: str = None,
                 label_path: str = None):
        
        self.feature_dir = feature_dir
        self.label_path = label_path
        self.n_batch = len(os.listdir(self.feature_dir))
        self.metadata = self.gen_metadata()
        self.feature_profiles = None
        
        if profile_path is not None:
            with open(profile_path, "r") as rf:
                self.feature_profiles = json.load(rf)
        
        self.labels = None
        self.col_name = None
        self.col_data = None
        self.sample_batches = None
        self.batch_data = None
        self.batch_labels = None
        
    def gen_metadata(self):
        metadata = init_metadata()
        pq_con = pq.ParquetDataset(self.feature_dir)
        columns = pq_con.schema.names
        metadata["features"] = [c for c in columns if 
                                c not in metadata["keys"]]
        metadata["continuous"] = [c for c in metadata["features"]
                                  if c not in metadata["cats"]]
        metadata["col_groups"] = {
            v: [c for c in metadata["features"] if c.startswith(k)]
            for k, v in metadata["col_types"].items()
        }
        return metadata
    
    def load_labels(self):
        self.labels = pd.read_csv(self.label_path)
    
    def load_batches(self, n_samples: int):
        
        batches = np.random.choice(range(self.n_batch), 
                                   n_samples, 
                                   replace=False)
        self.sample_batches = batches
        
        pq_filter = ("batch", "in", list(map(str, batches)))
        pq_con = pq.ParquetDataset(self.feature_dir, filters=[pq_filter])
        self.batch_data = pq_con.read().to_pandas().drop("batch", axis=1)
        
        customers = self.batch_data.index\
            .get_level_values("customer_ID").unique().tolist()
        
        if self.labels is None:
            self.load_labels()
        
        self.batch_labels = self.labels[self.labels["customer_ID"]\
            .isin(customers)].copy().set_index("customer_ID")
    
    def impute_binary_cat(self):
        if 0 not in self.col_data[self.col_name].unique():
                self.col_data.loc[:, self.col_name] = \
                    self.col_data[self.col_name].fillna(0).astype(int)
            
    def load_column(self, 
                    col_name: str, 
                    index_cols: list = None):   
        
        if index_cols is None:
            index_cols = self.metadata["keys"]
            
        pq_con = pq.ParquetDataset(self.feature_dir)
        self.col_name = col_name
        self.col_data = pq_con.read(columns=index_cols+[col_name])\
            .to_pandas().drop("batch", axis=1).reset_index()
        
        if self.col_name in self.metadata["cats"]:
            if self.col_data[self.col_name].isna().sum() > 0:
                self.impute_binary_cat()  
                
    def load_index(self, index_cols: list = None):
        
        if index_cols is None:
            index_cols = self.metadata["keys"]
            
        pq_con = pq.ParquetDataset(self.feature_dir)
        return pq_con.read(columns=index_cols)\
            .to_pandas().drop("batch", axis=1).reset_index()
        
        
if __name__ == "__main__":
    
    import sys
    sys.path.append(os.path.abspath("."))
    from amexdp import env_config
    env_config("config.json")
    
    dl = DataLoader(data_dir=os.environ.get("DATA_DIR"))
    dl.load_batches(3)
       
        