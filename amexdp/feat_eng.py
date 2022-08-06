
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Feature Engineering -+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import gc
from tqdm import tqdm
import pandas as pd
from amexdp.data_loader import DataLoader


def _default_value(x, default):
    if x is None:
        return default
    else:
        return x


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
    
    
class FeatureEngineering(DataLoader):
    
    def __init__(self, 
                 feature_dir: str, 
                 profile_path: str = None, 
                 label_path: str = None,
                 agg_default: list = None,
                 agg_custom: dict = None,
                 impute_default: list = None,
                 impute_custom: dict = None,
                 cat_default: list = None,
                 cat_custom: dict = None,
                 group_cols: list or str = "customer_ID"):
        
        super().__init__(feature_dir, profile_path, label_path)

        self.agg_default = _default_value(agg_default, ["last"])
        self.agg_custom = _default_value(agg_custom, {})
            
        self.impute_default = impute_default
        self.impute_custom = _default_value(impute_custom, {})
        
        self.cat_default = _default_value(cat_default, ["use_last_cat"])
        self.cat_custom = _default_value(cat_custom, {})   

        self.group_cols = group_cols
        self.agg_df = None
    
    def init_agg_frame(self,
                       add_labels: bool = False,
                       index_cols: list = None,
                       dt_col: str = "S_2"):
        
        raw_frame = self.load_index(index_cols=index_cols)
        
        if dt_col not in raw_frame.columns:
            raise Exception("dt_col must be in index_cols!")
        
        self.agg_df = raw_frame.groupby(self.group_cols)[[dt_col]].count()
        
        if add_labels:
            if self.labels is None:
                self.load_labels()
            self.agg_df = pd.concat([self.labels.set_index(self.group_cols), 
                                     self.agg_df], axis=1)
    
    def use_last_cat(self, df: pd.DataFrame, cat: str, append: bool = False):
              
        last_cat = aggregator(df, agg_spec="last", agg_cols=cat)
        cat_map_last = self.metadata["cat_maps"]["last"]
        
        if cat in cat_map_last.keys():
            last_cat.replace(cat_map_last[cat], inplace=True)
            
        if append:
            self.agg_df.loc[:, f"{cat}_last"] = last_cat.astype("category")
        else:
            return last_cat.rename(f"{cat}_last")
    
    def ohe_mean(self, df: pd.DataFrame, cat: str, append: bool = False):
        
        ohe_df = pd.get_dummies(self.col_data.set_index(self.group_cols)[cat]).\
            groupby(self.group_cols).mean().add_prefix(f"{cat}_OHE_")
        
        if append:
            self.agg_df = pd.concat([self.agg_df, ohe_df], axis=1)
        else:
            return ohe_df
    
    def gen_continuous_feature(self, col: str, append: bool = True):
        
        self.load_column(col)
        
        if col in self.agg_custom.keys():
            agg_spec = self.agg_custom[col]
        else:
            agg_spec = self.agg_default
        
        if col in self.impute_custom.keys():
            impute_args = self.impute_custom[col]
        else:
            impute_args = self.impute_default
            
        if impute_args is not None:
            if impute_args[1] == "before":
                self.col_data[col].fillna(impute_args[0], inplace=True)
                
        agg_data = aggregator(df=self.col_data,
                              agg_spec=agg_spec,
                              agg_cols=col)
        agg_data.columns = [f"{col}_{a}" for a in agg_spec]
        
        if impute_args is not None:
            if impute_args[1] == "after":
                agg_data.fillna(impute_args[0], inplace=True)
        
        if append:
            self.agg_df = pd.concat([self.agg_df, agg_data], axis=1)
        else:
            return agg_data
    
    def gen_categorical_feature(self, col: str, append: bool = True):
        
        self.load_column(col)
        cat_map_all = self.metadata["cat_maps"]["all"]
        
        if col in cat_map_all.keys():
            self.col_data.replace({col: cat_map_all[col]}, inplace=True)
        
        if col in self.cat_custom.keys():
            cat_procs = self.cat_custom[col]
        else:
            cat_procs = self.cat_default
            
        if append:
            for cp in cat_procs:
                getattr(self, cp)(df=self.col_data, cat=col, append=True)
        else:
            cat_df = []
            for cp in cat_procs:
                cat_df.append(getattr(self, cp)(df=self.col_data, cat=col))
            return pd.concat(cat_df)
        
    def run_feateng_pipeline(self,
                             cols: list = None,
                             training: bool = True,
                             progress: str = None):
        
        if cols is None:
            cols = self.metadata["features"]
        
        self.init_agg_frame(add_labels=training)
        
        if progress == "tqdm":
            cols = tqdm(cols)
        elif progress == "print":
            i = 0
        else:
            pass
           
        for col in cols:
            
            if col in self.metadata["continuous"]:
                self.gen_continuous_feature(col)
            elif col in self.metadata["cats"]:
                self.gen_categorical_feature(col)
            else:
                raise Exception(f"Feature {col} not in metadata!") 
            
            if progress == "print":
                print("Feature %s engineered (%d of %d)" % 
                      (col, i+1, len(cols)))
                i += 1
            
            _ = gc.collect()               
        