
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Data Loader (*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+

import os
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
        "cats": ['B_30', 'B_38', 'D_114', 'D_116', 
                 'D_117', 'D_120', 'D_126', 'D_63', 
                 'D_64', 'D_66', 'D_68']
    }
    
    return metadata

class DataLoader:

    def __init__(self, 
                 data_dir: str,
                 train_dir: str = "train_data",
                 label_file: str = "train_labels.csv"):
        
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, train_dir) 
        self.label_path = os.path.join(data_dir, label_file) 
        self.n_batch = len(os.listdir(self.train_path))
        self.metadata = self.gen_metadata()
        
        self.labels = None
        self.sample_batches = None
        self.batch_data = None
        self.batch_labels = None
    
    def gen_metadata(self):
        metadata = init_metadata()
        pq_con = pq.ParquetDataset(self.train_path)
        columns = pq_con.schema.names
        metadata["features"] = [c for c in columns if 
                                c not in metadata["keys"]]
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
        pq_con = pq.ParquetDataset(self.train_path, filters=[pq_filter])
        self.batch_data = pq_con.read().to_pandas().drop("batch", axis=1)
        
        customers = self.batch_data.index\
            .get_level_values("customer_ID").unique().tolist()
        
        if self.labels is None:
            self.load_labels()
        
        self.batch_labels = self.labels[self.labels["customer_ID"]\
            .isin(customers)].copy().set_index("customer_ID")
            
    def load_column(self, 
                    col_name: str, 
                    index_cols: list = None):   
        
        if index_cols is None:
            index_cols = self.metadata["keys"]
            
        pq_con = pq.ParquetDataset(self.train_path)
        return pq_con.read(columns=index_cols+[col_name])\
            .to_pandas().drop("batch", axis=1).reset_index()
        
if __name__ == "__main__":
    
    import sys
    sys.path.append(os.path.abspath("."))
    from amexdp import env_config
    env_config("config.json")
    
    dl = DataLoader(data_dir=os.environ.get("DATA_DIR"))
    dl.load_batches(3)
       
        