
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Data Profiler )+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import matplotlib.pyplot as plt
from amexdp.data_loader import DataLoader
from amexdp.feat_eng import aggregator
from amexdp.data_viz import bar_counter, signal_preview, auto_subplots


class DataProfiler(DataLoader):
    
    def __init__(self, 
                 feature_dir: str, 
                 profile_path: str = None,
                 label_path: str = None):
        
        super().__init__(feature_dir, profile_path, label_path)
        
        self.active_profiler = False
        self.default_agg = None
        self.hist_q = None
        self.n_cuts = None
        self.fig_width = None
        self.fig_height = None
        self.auto_subplots = True
        
    def init_profiler(self,
                      default_agg: dict = None,
                      hist_q: tuple = (0.005, 0.995),
                      n_cuts: int = 7,
                      figure_dims: tuple = (12, 3),
                      auto_subplots: bool = True):
        
        if default_agg is None:
            default_agg = {
                "continuous": ["mean", "min", "max", "last"],
                "categorical": ["nunique", "first", "last"]
            }
            
        self.active_profiler = True
        self.default_agg = default_agg
        self.hist_q = hist_q
        self.n_cuts = n_cuts
        self.fig_width, self.fig_height = figure_dims
        self.auto_subplots = auto_subplots
    
    def continous_profile(self, agg_list: list):
        
        col = self.col_name
        col_na = f"{col}_NA"
        df = self.col_data.copy()
        
        df.loc[:, col_na] = df[col].isna()
        agg_df = aggregator(df, agg_spec={col: agg_list, col_na: "mean"})
        agg_df.columns = agg_list + [col_na]
        agg_df = self.labels.join(agg_df, on="customer_ID")
        
        _, ax = plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
        
        df[col][(df[col] > df[col].quantile(self.hist_q[0])) & 
                (df[col] < df[col].quantile(self.hist_q[1]))].hist(ax=ax[0])
        ax[0].set_xlabel(col)
        
        agg_df.groupby(agg_df[col_na]>0)["target"].mean()\
            .plot(kind="barh", ax=ax[1])
        ax[1].set_title("Missing {} of {} ({}%)".format(
            agg_df[col_na].sum().round(), agg_df.shape[0],
            (agg_df[col_na].sum()/agg_df.shape[0]).round(4)*100
        ))
        ax[1].set_xlabel("default rate")
        plt.show()
        
        signal_preview(df=agg_df, cols=agg_list, continuous=True, 
                       cut_type="range", n_cuts=self.n_cuts,
                       width=self.fig_width, row_height=self.fig_height)
        
        signal_preview(df=agg_df, cols=agg_list, continuous=True, 
                       cut_type="quantile", n_cuts=self.n_cuts,
                       width=self.fig_width, row_height=self.fig_height)
        
        return agg_df
        
    def categorical_profile(self, agg_list: list):
        
        col = self.col_name
        df = self.col_data.copy()
        
        df.loc[:, col] = df[col].astype("category")
        agg_df = aggregator(df, agg_spec=agg_list, agg_cols=col)
        agg_df = self.labels.join(agg_df, on="customer_ID")
        
        if self.auto_subplots:
            ax_list = auto_subplots(n_plots=len(agg_list)+1,
                                    row_dims=(self.fig_width, self.fig_height))
        else:
            ax_list = [None]*(len(agg_list)+1)
        
        figsize = (self.fig_width, self.fig_height)
        
        bar_counter(df[col], figsize=figsize, ax=ax_list[0])
        for i, a in enumerate(agg_list):
            bar_counter(agg_df[a], figsize=figsize, ax=ax_list[i+1])
            
        if self.auto_subplots:
            plt.show()
            
        signal_preview(df=agg_df, cols=agg_list, continuous=False,
                       width=self.fig_width, row_height=self.fig_height)
        
        return agg_df
            
    def profile_column(self):
        
        if self.col_data is None:
            raise Exception("Run load_column() method first!")
        
        if self.labels is None:
            self.load_labels()
        
        if not self.active_profiler:
            self.init_profiler()
        
        if self.col_name in self.metadata["cats"]:
            return self.categorical_profile(self.default_agg["categorical"])
        else:
            return self.continous_profile(self.default_agg["continuous"])
        
    def quick_profile(self):
        
        if self.col_data is None:
            raise Exception("Run load_column() method first!")    
            
        if self.col_name in self.metadata["cats"]:
            
            return {
                "all": self.col_data[self.col_name].value_counts().to_dict(),
                "last": self.col_data.groupby("customer_ID")[self.col_name]\
                    .last().value_counts().to_dict()
            }
            
        else:
            
            summ_all = self.col_data[self.col_name]\
                .describe().round(4).to_dict()
            summ_all.update(
                {"missing": self.col_data.shape[0] - summ_all["count"]}
            )
            
            last = self.col_data.groupby("customer_ID")[self.col_name].last()
            summ_last = last.describe().round(4).to_dict()
            summ_last.update(
                {"missing": last.shape[0] - summ_last["count"]}
            )
            
            return {"all": summ_all, "last": summ_last}   

