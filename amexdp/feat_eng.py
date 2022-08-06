
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Feature Engineering -+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import pandas as pd
from amexdp.data_loader import DataLoader


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

        if agg_default is None:
            self.agg_default = ["last"]
        else:
            self.agg_default = agg_default
        self.agg_custom = agg_custom
            
        self.impute_default = impute_default
        self.impute_custom = impute_custom
        
        if cat_default is None:
            self.cat_default = ["use_last_cat"]
        else:
            self.cat_default = cat_default
        self.cat_custom = cat_custom   

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
            groupby(self.group_cols).mean().add_prefix(f"{cat}_")
        
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
        
    
        