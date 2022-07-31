
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Data Loader (*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import os
from importlib_metadata import metadata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


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


def aggregator(
    df: pd.DataFrame,
    agg_spec: str or list or dict,
    agg_cols: str or list = None,
    group_cols: str or list = "customer_ID"
):
    if agg_cols is None:
        return df.groupby(group_cols).agg(agg_spec)
    else:
        return df.groupby(group_cols)[agg_cols].agg(agg_spec)


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
        self.col_name = None
        self.col_data = None
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
        self.col_name = col_name
        self.col_data = pq_con.read(columns=index_cols+[col_name])\
            .to_pandas().drop("batch", axis=1).reset_index()
       
    def continous_profile(self,
                          agg_list: list,
                          hist_q: tuple = (0.005, 0.995)):
        
        col = self.col_name
        col_na = f"{col}_NA"
        df = self.col_data.copy()
        
        df.loc[:, col_na] = df[col].isna()
        agg_df = aggregator(df, agg_spec={col: agg_list, col_na: "mean"})
        agg_df = self.labels.join(agg_df, on="customer_ID")
        
        return agg_df
        
    def categorical_profile(self,
                            agg_list: list):
        
        col = self.col_name
        df = self.col_data.copy()
        
        df.loc[:, col] = df[col].astype("category")
        agg_df = aggregator(df, agg_spec=agg_list, agg_cols=col)
        agg_df = self.labels.join(agg_df, on="customer_ID")
        
        return agg_df
            
    def profile_column(self,
                       agg_list: list = None,
                       hist_q: tuple = (0.005, 0.995)):
        
        if self.col_data is None:
            raise Exception("Run load_column() method first!")
        
        if self.labels is None:
            self.load_labels()
        
        if self.col_name in self.metadata["cats"]:
            if agg_list is None:
                agg_list = ["nunique", "first", "last"]
            return self.categorical_profile(agg_list)
        else:
            if agg_list is None:
                agg_list = ["mean", "min", "max", "last"]
            return self.continous_profile(agg_list, hist_q)    
            
        
if __name__ == "__main__":
    
    import sys
    sys.path.append(os.path.abspath("."))
    from amexdp import env_config
    env_config("config.json")
    
    dl = DataLoader(data_dir=os.environ.get("DATA_DIR"))
    dl.load_batches(3)
       
        