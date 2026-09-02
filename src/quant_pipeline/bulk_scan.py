from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import ScanConfig
from .registry import FeatureSpec
from .scanner import benjamini_hochberg


@dataclass
class CudaFeatureContext:
    """Feature-side tensors reused across every target batch in a chunk."""
    feature_names: tuple[str, ...]
    device: object
    tx: object
    rx: object
    finite_x: object
    cluster_codes: object
    coverage_codes: dict[str, object]


@dataclass
class BinaryCudaContext:
    feature_names: tuple[str, ...]
    device: object
    finite_x: object
    x_on: object
    codes: dict[str, object]


def build_cuda_feature_context(
    feature_frame: pd.DataFrame,
    features: list[FeatureSpec],
    config: ScanConfig,
) -> CudaFeatureContext:
    import torch
    active=[s for s in features if s.classification!="categorical" and s.dtype not in {"categorical", "binary"}]
    device=torch.device(config.cuda_device if config.use_cuda and torch.cuda.is_available() else "cpu")
    names=tuple(s.name for s in active)
    x_np=feature_frame[list(names)].to_numpy(dtype=np.float32,copy=True)
    tx=torch.as_tensor(x_np,device=device)
    session_codes=pd.factorize(feature_frame.session_date,sort=False)[0]
    symbol_codes=pd.factorize(feature_frame.symbol,sort=False)[0]
    decisions=feature_frame.decision_ts if "decision_ts" in feature_frame else pd.Series(np.arange(len(feature_frame)),index=feature_frame.index)
    decision_codes=pd.factorize(decisions,sort=False)[0]
    years=pd.to_datetime(feature_frame.session_date).dt.year.to_numpy()
    year_codes=pd.factorize(years,sort=False)[0]
    codes={
        "sessions":torch.as_tensor(session_codes,device=device,dtype=torch.long),
        "symbols":torch.as_tensor(symbol_codes,device=device,dtype=torch.long),
        "decisions":torch.as_tensor(decision_codes,device=device,dtype=torch.long),
        "years":torch.as_tensor(year_codes,device=device,dtype=torch.long),
    }
    sample=np.linspace(0,len(feature_frame)-1,min(len(feature_frame),200_000),dtype=np.int64)
    rx=_percentile_bins(tx,x_np[sample],100)
    return CudaFeatureContext(names,device,tx,rx,torch.isfinite(tx),codes["sessions"],codes)


def build_cuda_binary_context(
    feature_frame: pd.DataFrame,
    features: list[FeatureSpec],
    config: ScanConfig,
) -> BinaryCudaContext:
    """Build feature and cluster tensors once for every target batch."""
    import torch
    device = torch.device(config.cuda_device if config.use_cuda and torch.cuda.is_available() else "cpu")
    names = tuple(item.name for item in features)
    eligible = feature_frame.analysis_eligible.fillna(False).to_numpy(bool) if "analysis_eligible" in feature_frame else np.ones(len(feature_frame), bool)
    x_np = feature_frame[list(names)].to_numpy(dtype=np.float64, copy=True)
    finite_x = torch.as_tensor(np.isfinite(x_np) & eligible[:, None], device=device)
    x_on = torch.as_tensor(np.nan_to_num(x_np, nan=0.0).astype(bool), device=device) & finite_x
    session_codes = pd.factorize(feature_frame.session_date.astype(str), sort=False)[0]
    symbol_codes = pd.factorize(feature_frame.symbol.astype(str), sort=False)[0]
    decision_codes = pd.factorize(feature_frame.decision_ts.astype(str), sort=False)[0]
    year_codes = pd.factorize(pd.to_datetime(feature_frame.session_date).dt.year, sort=False)[0]
    intersection_codes = session_codes * (int(symbol_codes.max()) + 1) + symbol_codes
    codes = {
        "sessions": torch.as_tensor(session_codes, device=device, dtype=torch.long),
        "symbols": torch.as_tensor(symbol_codes, device=device, dtype=torch.long),
        "decisions": torch.as_tensor(decision_codes, device=device, dtype=torch.long),
        "years": torch.as_tensor(year_codes, device=device, dtype=torch.long),
        "intersection": torch.as_tensor(intersection_codes, device=device, dtype=torch.long),
    }
    return BinaryCudaContext(names, device, finite_x, x_on, codes)


def cuda_screen(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    features: list[FeatureSpec],
    targets: list[str],
    config: ScanConfig,
    prior: pd.DataFrame,
    journal: Path,
    feature_context: CudaFeatureContext | None = None,
) -> pd.DataFrame:
    """Vectorized all-pair screen; exact diagnostics are reserved for survivors."""
    completed={(r.feature,r.target) for r in prior.itertuples()} if not prior.empty else set()
    active_features=[s for s in features if s.classification!="categorical" and s.dtype not in {"categorical", "binary"} and any((s.name,t) not in completed for t in targets)]
    if not active_features: return prior
    import torch
    feature_names=[s.name for s in active_features]
    owns_context=feature_context is None
    context=feature_context or build_cuda_feature_context(feature_frame,active_features,config)
    if tuple(feature_names)!=context.feature_names:
        raise ValueError("CUDA feature context does not match the active feature set")
    device=context.device; tx=context.tx; rx=context.rx
    y_np=target_frame[targets].to_numpy(dtype=np.float32,copy=True)
    ty=torch.as_tensor(y_np,device=device)
    corr,n,pair_mean_x,pair_mean_y,vx,vy,cov=_pair_moments_stable(tx,ty)
    cluster_se,cluster_t=_clustered_inference_from_moments(
        tx,ty,context.cluster_codes,n,pair_mean_x,pair_mean_y,vx,cov
    )
    # Approximate ranks on a deterministic 200k-row sample. Exact Spearman is
    # recomputed for promoted pairs in the diagnostic pass.
    sample=np.linspace(0,len(feature_frame)-1,min(len(feature_frame),200_000),dtype=np.int64)
    ry=_percentile_bins(ty,y_np[sample],100)
    rank_corr,*_=_pair_moments_stable(rx,ry)
    coverage=_pair_coverage_counts(
        context.finite_x,torch.isfinite(ty),context.coverage_codes,
    )
    rows=[]
    corr=corr.cpu().numpy(); rank_corr=rank_corr.cpu().numpy(); n=n.cpu().numpy(); pair_mean_y=pair_mean_y.cpu().numpy(); vx=vx.cpu().numpy(); vy=vy.cpu().numpy(); cov=cov.cpu().numpy(); cluster_se=cluster_se.cpu().numpy(); cluster_t=cluster_t.cpu().numpy()
    # Aggregate all ten deciles for every target in one scatter operation per
    # feature. This replaces thousands of full-column masks and device syncs.
    decile_means=[]; decile_counts=[]
    finite_y=torch.isfinite(ty)
    for i in range(len(active_features)):
        feature_valid=torch.isfinite(tx[:,i]); bins=(rx[:,i].nan_to_num(0).clamp(0,99)//10).long()
        feature_means=np.full((10,ty.shape[1]),np.nan,dtype=np.float32); feature_counts=np.zeros((10,ty.shape[1]),dtype=np.int64)
        for j in range(ty.shape[1]):
            valid=feature_valid&finite_y[:,j]; valid_bins=bins[valid]
            counts=torch.bincount(valid_bins,minlength=10)
            sums=torch.zeros(10,device=device,dtype=ty.dtype); sums.scatter_add_(0,valid_bins,ty[valid,j])
            counts_np=counts.cpu().numpy(); feature_counts[:,j]=counts_np
            feature_means[:,j]=np.where(counts_np>0,(sums/counts.clamp_min(1)).cpu().numpy(),np.nan)
        decile_means.append(feature_means); decile_counts.append(feature_counts)
    for i,spec in enumerate(active_features):
        for j,target in enumerate(targets):
            if (spec.name,target) in completed: continue
            count=int(n[i,j])
            sessions=int(coverage["sessions"][i,j]); symbols=int(coverage["symbols"][i,j]); decisions=int(coverage["decisions"][i,j]); valid_years=int(coverage["years"][i,j])
            base={"feature":spec.name,"feature_family":spec.family,"feature_classification":spec.classification,"target":target,"target_family":target.removesuffix("_benchmark_adjusted"),"table_rows":len(feature_frame),"n":count,"valid_observations":count,"sessions":sessions,"valid_sessions":sessions,"symbols":symbols,"valid_symbols":symbols,"valid_decision_timestamps":decisions,"valid_years":valid_years}
            pair_variance=float(vx[i,j]/max(count,1))
            if not np.isfinite(pair_variance) or pair_variance<1e-14:
                rows.append({**base,"status":"constant_feature"}); continue
            if count<config.min_observations or sessions<config.min_sessions or symbols<config.min_symbols or decisions<config.min_decision_timestamps or valid_years<config.min_years:
                rows.append({**base,"status":"insufficient_data"}); continue
            counts=decile_counts[i][:,j].tolist(); means=decile_means[i][:,j].astype(float)
            means=np.where(np.asarray(counts)>0,means,np.nan)
            finite=np.isfinite(means)
            monotonicity=float(stats.spearmanr(np.arange(10)[finite],np.asarray(means)[finite]).statistic) if finite.sum()>=3 else np.nan
            spread=float(means[-1]-means[0]) if np.isfinite(means[-1]) and np.isfinite(means[0]) else np.nan
            r=float(corr[i,j]); t=float(cluster_t[i,j]); p=float(2*stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan
            variance_x=vx[i,j]; slope=float(cov[i,j]/variance_x) if variance_x>0 else np.nan
            mean_y=float(pair_mean_y[i,j]); var_y=max(0,float(vy[i,j]/count))
            shape="positive_monotonic" if monotonicity>=.7 else "negative_monotonic" if monotonicity<=-.7 else "no_stable_shape"
            rows.append({**base,"mean_target":mean_y,"std_target":sqrt(var_y),"pearson":r,"spearman":float(rank_corr[i,j]),"slope":slope,"cluster_se":float(cluster_se[i,j]),"cluster_t":t,"raw_p":p,"top_bottom_spread":spread,"monotonicity":monotonicity,"shape":shape,"status":"cuda_screened"})
    additions=pd.DataFrame(rows)
    if not additions.empty:
        assert_valid_screen_results(additions,"CUDA screen batch")
        additions.to_json(journal,orient="records",lines=True,mode="a",double_precision=15)
    out=pd.concat([prior,additions],ignore_index=True)
    del ty,ry
    if owns_context:
        del context
        if device.type=="cuda": torch.cuda.empty_cache()
    return out


def cuda_binary_scan_batch(
    frame: pd.DataFrame,
    features: list[FeatureSpec],
    targets: list[str],
    config: ScanConfig,
    context: BinaryCudaContext | None = None,
) -> pd.DataFrame:
    """Exact binary on/off effects with batched two-way cluster inference on CUDA."""
    if not features:
        return pd.DataFrame()
    import torch

    owns_context = context is None
    context = context or build_cuda_binary_context(frame, features, config)
    if tuple(item.name for item in features) != context.feature_names:
        raise ValueError("CUDA binary feature context does not match the active feature set")
    device = context.device
    finite_x = context.finite_x
    x_on = context.x_on
    codes = context.codes

    def grouped(
        valid: torch.Tensor,
        on: torch.Tensor,
        y: torch.Tensor,
        name: str,
        beta0: torch.Tensor,
        beta1: torch.Tensor,
        *,
        covariance: bool,
    ):
        group_codes = codes[name]
        groups = int(group_codes.max().item()) + 1
        width = valid.shape[1]
        counts = torch.zeros((groups, width), device=device, dtype=torch.float64)
        on_counts = torch.zeros_like(counts)
        counts.index_add_(0, group_codes, valid.to(torch.float64))
        on_counts.index_add_(0, group_codes, on.to(torch.float64))
        valid_groups = (counts > 0).sum(0)
        on_groups = (on_counts > 0).sum(0)
        off_groups = ((counts - on_counts) > 0).sum(0)
        if not covariance:
            return None, valid_groups, on_groups, off_groups
        sum_y = torch.zeros_like(counts)
        sum_xy = torch.zeros_like(counts)
        sum_y.index_add_(0, group_codes, torch.where(valid, y[:, None], 0.0))
        sum_xy.index_add_(0, group_codes, torch.where(on, y[:, None], 0.0))
        score0 = sum_y - beta0 * counts - beta1 * on_counts
        score1 = sum_xy - (beta0 + beta1) * on_counts
        meat00 = (score0 * score0).sum(0)
        meat01 = (score0 * score1).sum(0)
        meat11 = (score1 * score1).sum(0)
        return (meat00, meat01, meat11), valid_groups, on_groups, off_groups

    rows = []
    for target in targets:
        y_np = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float64)
        y = torch.as_tensor(np.nan_to_num(y_np, nan=0.0), device=device)
        finite_y = torch.as_tensor(np.isfinite(y_np), device=device)
        valid = finite_x & finite_y[:, None]
        on = x_on & finite_y[:, None]
        n = valid.sum(0).to(torch.float64)
        n_on = on.sum(0).to(torch.float64)
        n_off = n - n_on
        sum_y = torch.where(valid, y[:, None], 0.0).sum(0)
        sum_y2 = torch.where(valid, y[:, None] * y[:, None], 0.0).sum(0)
        sum_xy = torch.where(on, y[:, None], 0.0).sum(0)
        on_mean = sum_xy / n_on.clamp_min(1)
        off_mean = (sum_y - sum_xy) / n_off.clamp_min(1)
        beta0 = off_mean
        beta1 = on_mean - off_mean
        date_meat, sessions, on_sessions, off_sessions = grouped(valid, on, y, "sessions", beta0, beta1, covariance=True)
        symbol_meat, symbols, on_symbols, off_symbols = grouped(valid, on, y, "symbols", beta0, beta1, covariance=True)
        intersection_meat, intersection_groups, _, _ = grouped(valid, on, y, "intersection", beta0, beta1, covariance=True)
        _, decisions, on_decisions, off_decisions = grouped(valid, on, y, "decisions", beta0, beta1, covariance=False)
        _, years, _, _ = grouped(valid, on, y, "years", beta0, beta1, covariance=False)

        def slope_covariance(meat, group_count):
            meat00, meat01, meat11 = meat
            r0 = -1.0 / n_off.clamp_min(1)
            r1 = 1.0 / n_on.clamp_min(1) + 1.0 / n_off.clamp_min(1)
            variance = r0 * r0 * meat00 + 2.0 * r0 * r1 * meat01 + r1 * r1 * meat11
            correction = (group_count / (group_count - 1).clamp_min(1)) * ((n - 1) / (n - 2).clamp_min(1))
            return variance * correction

        date_variance = slope_covariance(date_meat, sessions.to(torch.float64))
        symbol_variance = slope_covariance(symbol_meat, symbols.to(torch.float64))
        intersection_variance = slope_covariance(intersection_meat, intersection_groups.to(torch.float64))
        variance = torch.clamp(date_variance + symbol_variance - intersection_variance, min=0.0)
        se = torch.sqrt(variance)
        t_value = beta1 / se
        probabilities = torch.erfc(torch.abs(t_value) / np.sqrt(2.0))
        mean_y = sum_y / n.clamp_min(1)
        var_y = torch.clamp(sum_y2 / n.clamp_min(1) - mean_y * mean_y, min=0.0)
        var_x = n_on * n_off / n.clamp_min(1)
        covariance_xy = sum_xy - n_on * mean_y
        pearson = covariance_xy / torch.sqrt(var_x * (sum_y2 - sum_y * mean_y)).clamp_min(1e-30)
        values = {
            "n": n, "n_on": n_on, "n_off": n_off, "sessions": sessions, "symbols": symbols,
            "decisions": decisions, "years": years, "on_sessions": on_sessions, "off_sessions": off_sessions,
            "on_symbols": on_symbols, "off_symbols": off_symbols, "on_decisions": on_decisions,
            "off_decisions": off_decisions, "on_mean": on_mean, "off_mean": off_mean, "effect": beta1,
            "mean_y": mean_y, "var_y": var_y, "pearson": pearson, "se": se, "t": t_value, "p": probabilities,
        }
        values = {name: tensor.cpu().numpy() for name, tensor in values.items()}
        for index, spec in enumerate(features):
            count = int(values["n"][index]); session_count = int(values["sessions"][index]); symbol_count = int(values["symbols"][index]); decision_count = int(values["decisions"][index]); year_count = int(values["years"][index])
            base = {"feature": spec.name, "feature_family": spec.family, "feature_classification": spec.classification, "scan_kind": "binary", "discovery_phase": spec.discovery_phase, "arity": spec.arity, "operator": spec.operator, "parent_features": str(spec.parent_features), "redundancy_group": spec.redundancy_group, "target": target, "target_family": target.removesuffix("_benchmark_adjusted"), "table_rows": len(frame), "n": count, "valid_observations": count, "sessions": session_count, "valid_sessions": session_count, "symbols": symbol_count, "valid_symbols": symbol_count, "valid_decision_timestamps": decision_count, "valid_years": year_count, "raw_p": np.nan}
            sufficient = count >= config.min_observations and session_count >= config.min_sessions and symbol_count >= config.min_symbols and decision_count >= config.min_decision_timestamps and year_count >= config.min_years
            state_sufficient = values["n_on"][index] >= config.binary_min_on_observations and values["n_off"][index] >= config.binary_min_off_observations and values["on_sessions"][index] >= config.binary_min_on_sessions and values["off_sessions"][index] >= config.binary_min_off_sessions and values["on_symbols"][index] >= config.binary_min_on_symbols and values["off_symbols"][index] >= config.binary_min_off_symbols
            if not sufficient or not state_sufficient:
                rows.append({**base, "status": "insufficient_data"}); continue
            effect = float(values["effect"][index]); standard_error = float(values["se"][index]); probability = float(np.clip(values["p"][index], 0.0, 1.0))
            rows.append({**base, "on_count": int(values["n_on"][index]), "off_count": int(values["n_off"][index]), "signal_on_count": int(values["n_on"][index]), "signal_off_count": int(values["n_off"][index]), "signal_on_sessions": int(values["on_sessions"][index]), "signal_off_sessions": int(values["off_sessions"][index]), "signal_on_symbols": int(values["on_symbols"][index]), "signal_off_symbols": int(values["off_symbols"][index]), "signal_on_decision_timestamps": int(values["on_decisions"][index]), "signal_off_decision_timestamps": int(values["off_decisions"][index]), "activation_rate": float(values["n_on"][index] / max(count, 1)), "on_mean_target": float(values["on_mean"][index]), "off_mean_target": float(values["off_mean"][index]), "mean_target": float(values["mean_y"][index]), "median_target": np.nan, "pearson": float(values["pearson"][index]), "spearman": np.nan, "slope": effect, "cluster_se": np.nan, "cluster_t": np.nan, "date_cluster_p": np.nan, "two_way_cluster_se": standard_error, "two_way_cluster_t": float(values["t"][index]), "two_way_cluster_p": probability, "raw_p": probability, "screen_inference": "two_way_date_symbol_cuda_exact_binary", "top_bottom_spread": effect, "monotonicity": np.nan, "shape": "binary_positive" if effect > 0 else "binary_negative", "effect_kind": "binary_on_minus_off", "effect_value": effect, "status": "binary_screened"})
    if owns_context and device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _pair_coverage_counts(finite_x,finite_y,group_codes:dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """Count distinct valid groups for every feature-target pair on device."""
    import torch
    features=finite_x.shape[1]; targets=finite_y.shape[1]; device=finite_x.device
    codes={name:(values if torch.is_tensor(values) else torch.as_tensor(values,device=device,dtype=torch.long)) for name,values in group_codes.items()}
    sizes={name:int(values.max().item())+1 if values.numel() else 0 for name,values in codes.items()}
    counts={name:torch.zeros((features,targets),device=device,dtype=torch.int64) for name in codes}
    for target in range(targets):
        valid=(finite_x&finite_y[:,target,None]).to(torch.uint8)
        for name,values in codes.items():
            seen=torch.zeros((sizes[name],features),device=device,dtype=torch.uint8)
            seen.scatter_reduce_(0,values[:,None].expand(-1,features),valid,reduce="amax",include_self=True)
            counts[name][:,target]=seen.sum(0)
    return {name:values.cpu().numpy() for name,values in counts.items()}


def finalize_screen(result: pd.DataFrame) -> pd.DataFrame:
    assert_valid_screen_results(result,"screen before FDR")
    result=_finalize(result)
    assert_valid_screen_results(result,"screen after FDR",check_fdr=True)
    return result


def assert_valid_screen_results(result:pd.DataFrame,context:str,check_fdr:bool=False)->None:
    """Fail before promotion when persisted or computed screen statistics are impossible."""
    if result.empty:return
    if {"feature","target"}.issubset(result) and result.duplicated(["feature","target"]).any():
        raise RuntimeError(f"Duplicate feature-target rows in {context}")
    probability_columns=["raw_p"]
    if check_fdr:probability_columns += [c for c in ["bh_fdr_p","bh_fdr_p_global","bh_fdr_p_group","primary_global_fdr","family_fdr","cluster_fdr","exploratory_family_fdr"] if c in result]
    for column in probability_columns:
        if column not in result:continue
        values=pd.to_numeric(result[column],errors="coerce").dropna()
        invalid=values.lt(0)|values.gt(1)|~np.isfinite(values)
        if invalid.any():raise RuntimeError(f"Invalid probability in {context}: {column}={values.loc[invalid].iloc[0]}")
    for column in ["n","valid_observations","sessions","valid_sessions","symbols","valid_symbols","valid_decision_timestamps","valid_years"]:
        if column in result and pd.to_numeric(result[column],errors="coerce").dropna().lt(0).any():raise RuntimeError(f"Negative coverage count in {context}: {column}")


def _pair_moments(x,y):
    import torch
    mx=torch.isfinite(x).float(); my=torch.isfinite(y).float(); x0=torch.nan_to_num(x); y0=torch.nan_to_num(y)
    n=mx.T@my; sx=x0.T@my; sy=mx.T@y0; sxx=(x0*x0).T@my; syy=mx.T@(y0*y0); sxy=x0.T@y0
    cov=sxy-sx*sy/n.clamp_min(1); vx=sxx-sx*sx/n.clamp_min(1); vy=syy-sy*sy/n.clamp_min(1); corr=cov/torch.sqrt((vx*vy).clamp_min(1e-20))
    return corr,n,sx,sy,sxx,syy,sxy


def _pair_moments_stable(x,y,feature_block: int = 8):
    """Two-pass pairwise moments, stable when target availability changes feature variance."""
    import torch
    features=x.shape[1]; targets=y.shape[1]; shape=(features,targets)
    corr=torch.full(shape,float("nan"),device=x.device,dtype=x.dtype); n=torch.zeros(shape,device=x.device,dtype=x.dtype)
    mean_x=torch.full(shape,float("nan"),device=x.device,dtype=x.dtype); mean_y=torch.full(shape,float("nan"),device=x.device,dtype=x.dtype); vx=torch.zeros(shape,device=x.device,dtype=x.dtype); vy=torch.zeros(shape,device=x.device,dtype=x.dtype); cov=torch.zeros(shape,device=x.device,dtype=x.dtype)
    for j in range(targets):
        ycol=y[:,j]
        for start in range(0,features,feature_block):
            stop=min(features,start+feature_block); xb=x[:,start:stop]
            valid=torch.isfinite(xb)&torch.isfinite(ycol[:,None]); counts=valid.sum(0).to(x.dtype).clamp_min(1)
            mx=torch.where(valid,xb,0).sum(0)/counts; my=torch.where(valid,ycol[:,None],0).sum(0)/counts
            dx=torch.where(valid,xb-mx,0); dy=torch.where(valid,ycol[:,None]-my,0)
            block_vx=dx.square().sum(0); block_vy=dy.square().sum(0); block_cov=(dx*dy).sum(0)
            n[start:stop,j]=counts; mean_x[start:stop,j]=mx; mean_y[start:stop,j]=my; vx[start:stop,j]=block_vx; vy[start:stop,j]=block_vy; cov[start:stop,j]=block_cov
            corr[start:stop,j]=(block_cov/torch.sqrt((block_vx*block_vy).clamp_min(1e-20))).clamp(-1,1)
    return corr,n,mean_x,mean_y,vx,vy,cov


def _clustered_inference_from_moments(x,y,clusters,n,mean_x,mean_y,vx,cov,feature_block: int = 8):
    """One-way session-clustered OLS slope errors for every feature-target pair.

    The previous CUDA screen substituted the number of sessions into an IID
    correlation t statistic. That is not a clustered standard error and can be
    dramatically more conservative than the exact sandwich calculation.  For
    simple OLS with an intercept, each cluster's slope influence is
    ``sum((x-xbar) * residual)``.  Accumulating those scores gives the exact
    one-way cluster sandwich variance without materializing a regression per
    pair.
    """
    import torch

    shape=(x.shape[1],y.shape[1])
    standard_error=torch.full(shape,float("nan"),device=x.device,dtype=x.dtype)
    t_stat=torch.full(shape,float("nan"),device=x.device,dtype=x.dtype)
    cluster_count=int(clusters.max().item())+1 if clusters.numel() else 0
    for j in range(y.shape[1]):
        ycol=y[:,j]
        for start in range(0,x.shape[1],feature_block):
            stop=min(x.shape[1],start+feature_block); xb=x[:,start:stop]
            valid=torch.isfinite(xb)&torch.isfinite(ycol[:,None])
            mx=mean_x[start:stop,j]; my=mean_y[start:stop,j]
            block_vx=vx[start:stop,j]; slope=cov[start:stop,j]/block_vx.clamp_min(1e-20)
            dx=torch.where(valid,xb-mx,0); dy=torch.where(valid,ycol[:,None]-my,0)
            influence=dx*(dy-dx*slope)
            cluster_scores=torch.zeros((cluster_count,stop-start),device=x.device,dtype=x.dtype)
            cluster_scores.index_add_(0,clusters,influence)
            se=torch.sqrt(cluster_scores.square().sum(0).clamp_min(0))/block_vx.clamp_min(1e-20)
            valid_result=(n[start:stop,j]>2)&torch.isfinite(se)&se.gt(0)&block_vx.gt(1e-14)
            standard_error[start:stop,j]=torch.where(valid_result,se,torch.nan)
            t_stat[start:stop,j]=torch.where(valid_result,slope/se,torch.nan)
    return standard_error,t_stat


def _percentile_bins(tensor, sample: np.ndarray, bins: int):
    import torch
    out=torch.full_like(tensor,float("nan")); quantiles=np.linspace(0,1,bins+1)[1:-1]
    for column in range(tensor.shape[1]):
        values=sample[:,column]; values=values[np.isfinite(values)]
        if not len(values): continue
        edges=torch.as_tensor(np.unique(np.quantile(values,quantiles)),device=tensor.device,dtype=tensor.dtype)
        valid=torch.isfinite(tensor[:,column]); out[valid,column]=torch.bucketize(tensor[valid,column],edges).float()
    return out


def _finalize(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:return result
    result["bh_fdr_p_group"]=result.groupby(["feature_family","target_family"],dropna=False).raw_p.transform(benjamini_hochberg)
    result["bh_fdr_p_global"]=benjamini_hochberg(result.raw_p)
    # Global FDR is the promotion gate; grouped FDR remains diagnostic only.
    result["bh_fdr_p"]=result["bh_fdr_p_global"]
    eligible=result.raw_p.notna()&~result.status.isin(["constant_feature","insufficient_data"])
    result.loc[eligible,"status"]="no_meaningful_relationship"
    result.loc[eligible&(result.bh_fdr_p_global<.05),"status"]="statistically_interesting"
    result["test_count"]=int(result.raw_p.notna().sum()); effect=result.top_bottom_spread.abs().fillna(0).rank(pct=True); significance=(1-result.bh_fdr_p.fillna(1)).clip(0,1)
    result["anomaly_score"]=.4*effect+.35*significance+.25*result.monotonicity.abs().fillna(0)
    derived_group=result.feature.str.replace(r"_(1|2|3|4|5|6|8|10|12|15|20|24|30|36|48|60|78)$","_LOOKBACK",regex=True)
    if "redundancy_group" not in result:result["redundancy_group"]=derived_group
    else:
        supplied=result["redundancy_group"].astype("string")
        result["redundancy_group"]=supplied.where(supplied.notna()&supplied.str.len().gt(0),derived_group)
    return result.sort_values("anomaly_score",ascending=False,na_position="last").reset_index(drop=True)
