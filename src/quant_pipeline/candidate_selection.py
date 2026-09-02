from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScanConfig
from .scanner import benjamini_hochberg


def target_horizon_family(target:str)->str:
    if "eod" in target:return "long_intraday"
    match=re.search(r"_(\d+)m",target); horizon=int(match.group(1)) if match else 999
    if horizon<=15:return "very_short"
    if horizon<=45:return "short"
    if horizon<=120:return "medium"
    return "long_intraday"


def select_exact_candidate_queue(results:pd.DataFrame,config:ScanConfig,feature_paths:dict[str,Path]|None=None,limit:int=250)->pd.DataFrame:
    if results.empty:return results
    eligible=results.loc[results.raw_p.notna()].sort_values(["primary_global_fdr","anomaly_score"],ascending=[True,False])
    ranked=pd.concat([eligible.head(100),eligible.loc[eligible.primary_global_fdr.lt(.10)]]).drop_duplicates(["feature","target"]).copy()
    if ranked.empty:return ranked
    ranked=ranked.groupby("feature_family",sort=False).head(config.max_candidates_per_feature_family)
    features=ranked.feature.drop_duplicates().tolist(); parent={name:name for name in features}
    def find(name):
        while parent[name]!=name:parent[name]=parent[parent[name]]; name=parent[name]
        return name
    def union(left,right):
        a,b=find(left),find(right)
        if a!=b:parent[b]=a
    metadata=ranked.drop_duplicates("feature").set_index("feature")
    for _,group in metadata.groupby("feature_family"):
        by_redundancy={}
        for name in group.index:by_redundancy.setdefault(group.loc[name,"redundancy_group"],[]).append(name)
        for related in by_redundancy.values():
            for name in related[1:]:union(related[0],name)
    if feature_paths:
        samples={};by_path={}
        for feature in features:
            if feature in feature_paths:by_path.setdefault(feature_paths[feature],[]).append(feature)
        for path,names in by_path.items():
            frame=pd.read_parquet(path,columns=names);step=max(1,len(frame)//20_000);frame=frame.iloc[::step]
            for name in names:samples[name]=frame[name].reset_index(drop=True)
        for _,group in metadata.groupby("feature_family"):
            names=[name for name in group.index if name in samples]
            if len(names)<2:continue
            correlations=pd.concat({name:samples[name] for name in names},axis=1).corr(method="spearman")
            for index,left in enumerate(names):
                for right in names[index+1:]:
                    value=correlations.loc[left,right]
                    if pd.notna(value) and abs(value)>=.85:union(left,right)
    ranked["feature_cluster"]=ranked.feature.map(find)
    monotonicity=ranked.get("monotonicity",pd.Series(np.nan,index=ranked.index,dtype=float))
    spearman=ranked.get("spearman",pd.Series(np.nan,index=ranked.index,dtype=float))
    response_sign=np.sign(monotonicity.fillna(spearman).fillna(0)).astype(int).astype(str)
    ranked["candidate_cluster"]=ranked.feature_cluster+"__"+ranked.target.map(target_horizon_family)+"__response_"+response_sign
    ranked["cluster_fdr"]=ranked.groupby("candidate_cluster",dropna=False).raw_p.transform(benjamini_hochberg)
    representatives=[]
    for _,group in ranked.groupby("candidate_cluster",sort=False):
        best=group.head(1)
        simplest=group.assign(complexity=group.feature.str.count("_")+group.feature.str.extract(r"_(\d+)(?:m)?$",expand=False).fillna("0").astype(int)/1000).sort_values("complexity").head(1)
        neighbor=group.iloc[[min(1,len(group)-1)]]
        representatives.append(pd.concat([best,simplest,neighbor]).drop_duplicates(["feature","target"]).head(config.max_candidates_per_cluster))
    promoted=pd.concat(representatives,ignore_index=True).drop_duplicates(["feature","target"])
    return promoted.groupby(promoted.target.map(target_horizon_family),sort=False).head(config.max_candidates_per_target_family).head(limit)
