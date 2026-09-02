from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def gpu_dense_pair(feature_path: Path, target_path: Path, feature: str, target: str, min_bin: int,selection_end:str, scan_kind: str="continuous") -> tuple[dict, list[dict]]:
    """Exact dense pair statistics on CUDA; grouped diagnostics stay on CPU."""
    import torch

    ff=pd.read_parquet(feature_path,columns=["session_date","analysis_eligible",feature]); mask=pd.to_datetime(ff.session_date).le(pd.Timestamp(selection_end))&ff.analysis_eligible.fillna(False)
    x=ff.loc[mask,feature].to_numpy(dtype=np.float32,copy=False)
    y=pd.read_parquet(target_path,columns=[target]).loc[mask,target].to_numpy(dtype=np.float32,copy=False)
    valid=np.isfinite(x)&np.isfinite(y); x=x[valid]; y=y[valid]
    if scan_kind == "binary":
        on=y[x == 1]; off=y[x == 0]
        if not len(on) or not len(off):
            return {"pearson":np.nan,"spearman":np.nan,"top_bottom_spread":np.nan,"monotonicity":np.nan,"shape":"insufficient"}, []
        effect=float(on.mean()-off.mean())
        records=[{"signal":0,"count":int(len(off)),"mean":float(off.mean()),"median":float(np.median(off)),"win_rate":float((off>0).mean())},{"signal":1,"count":int(len(on)),"mean":float(on.mean()),"median":float(np.median(on)),"win_rate":float((on>0).mean())}]
        return {"pearson":float(np.corrcoef(x,y)[0,1]),"spearman":float(stats.spearmanr(x,y).statistic),"outlier_sensitivity":np.nan,"top_bottom_spread":effect,"monotonicity":np.nan,"shape":"binary_positive" if effect > 0 else "binary_negative","mean_target":float(y.mean()),"median_target":float(np.median(y)),"std_target":float(y.std()),"win_rate":float((y>0).mean()),"skewness":float(stats.skew(y)),"downside_p05":float(np.quantile(y,.05)),"upside_p95":float(np.quantile(y,.95)),"effect_kind":"binary_on_minus_off","effect_value":effect}, records
    device=torch.device("cuda:0"); tx=torch.as_tensor(x,device=device); ty=torch.as_tensor(y,device=device)
    pearson=float(_corr(tx,ty).item())
    rx,order=_average_ranks(tx); ry,_=_average_ranks(ty); spearman=float(_corr(rx,ry).item())
    bounds=torch.quantile(tx,torch.tensor([.01,.99],device=device)); clipped=tx.clamp(bounds[0],bounds[1]); cr,_=_average_ranks(clipped)
    outlier_sensitivity=abs(spearman-float(_corr(cr,ry).item()))

    # pandas qcut(rank(method="first"), 10) is an ordinal equal-count split.
    bins=torch.empty(len(tx),device=device,dtype=torch.int64)
    bins[order]=torch.clamp(torch.arange(len(tx),device=device)*10//max(1,len(tx)),max=9)
    records=[]; means=[]
    for q in range(10):
        values=ty[bins.eq(q)]; count=int(values.numel())
        if count:
            mean=float(values.mean().item()); std=float(values.std(unbiased=True).item()) if count>1 else np.nan
            median=float(torch.quantile(values,.5).item()); win=float(values.gt(0).float().mean().item()); se=std/np.sqrt(count) if count>1 else np.nan
        else: mean=median=win=std=se=np.nan
        means.append(mean); records.append({"bin":q,"count":count,"mean":mean,"median":median,"win_rate":win,"std":std,"se":se,"ci_low":mean-1.96*se,"ci_high":mean+1.96*se})
    finite=np.isfinite(means); monotonicity=float(stats.spearmanr(np.arange(10)[finite],np.asarray(means)[finite]).statistic) if finite.sum()>=3 else np.nan
    spread=float(means[-1]-means[0]) if np.isfinite(means[-1]) and np.isfinite(means[0]) else np.nan
    shape=_shape(np.asarray(means),monotonicity,min(r["count"] for r in records),min_bin)
    mean_y=ty.mean(); centered=ty-mean_y; variance=(centered*centered).mean(); skew=(centered**3).mean()/variance.clamp_min(1e-20).pow(1.5)
    quantiles=torch.quantile(ty,torch.tensor([.05,.5,.95],device=device))
    dense={"pearson":pearson,"spearman":spearman,"outlier_sensitivity":outlier_sensitivity,"top_bottom_spread":spread,"monotonicity":monotonicity,"shape":shape,"mean_target":float(mean_y.item()),"median_target":float(quantiles[1].item()),"std_target":float(ty.std(unbiased=True).item()),"win_rate":float(ty.gt(0).float().mean().item()),"skewness":float(skew.item()),"downside_p05":float(quantiles[0].item()),"upside_p95":float(quantiles[2].item())}
    del tx,ty,rx,ry,cr,clipped,bins
    torch.cuda.empty_cache()
    return dense,records


def _average_ranks(values):
    import torch
    order=torch.argsort(values,stable=True); sorted_values=values[order]
    starts=torch.ones(len(values),device=values.device,dtype=torch.bool)
    if len(values)>1: starts[1:]=sorted_values[1:].ne(sorted_values[:-1])
    groups=starts.cumsum(0)-1; counts=torch.bincount(groups); first=counts.cumsum(0)-counts
    averages=first.to(values.dtype)+(counts.to(values.dtype)-1)/2+1
    ranked=torch.empty_like(values); ranked[order]=averages[groups]
    return ranked,order


def _corr(x,y):
    import torch
    centered_x=x-x.mean(); centered_y=y-y.mean()
    return (centered_x*centered_y).sum()/torch.sqrt((centered_x.square().sum()*centered_y.square().sum()).clamp_min(1e-20))


def _shape(means: np.ndarray, rho: float, minimum_count: int, min_bin: int) -> str:
    if len(means)<3 or minimum_count<min_bin:return "insufficient"
    diffs=np.diff(means)
    if abs(rho)>=.7:return "positive_monotonic" if rho>0 else "negative_monotonic"
    if max(means[0],means[-1])-means[len(means)//2]>.5*abs(means[-1]-means[0]):return "two_sided_tail"
    if means[len(means)//2]>max(means[0],means[-1]):return "inverted_u"
    if means[len(means)//2]<min(means[0],means[-1]):return "u_shaped"
    if abs(diffs[-1])>2*np.nanmedian(abs(diffs[:-1])):return "positive_tail" if diffs[-1]>0 else "negative_tail"
    return "no_stable_shape"
