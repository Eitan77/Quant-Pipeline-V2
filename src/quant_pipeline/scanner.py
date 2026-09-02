from __future__ import annotations

from dataclasses import asdict
import json
from math import erf, sqrt

import numpy as np
import pandas as pd
from scipy import stats

from .config import ScanConfig
from .registry import FeatureSpec
from .gpu import CorrelationBackend
from .effects import feature_scan_kind
from .binary_coverage import binary_coverage, pair_status


def _normal_pvalue(z: float) -> float:
    return float(np.clip(1-erf(abs(z)/sqrt(2)),0,1))


def _clustered_slope(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float, float]:
    """OLS slope with one-way session-cluster sandwich error, no IID claim."""
    X=np.column_stack((np.ones(len(x)),x)); beta=np.linalg.pinv(X.T@X)@(X.T@y); resid=y-X@beta
    bread=np.linalg.pinv(X.T@X)
    codes,_=pd.factorize(clusters,sort=False)
    score0=np.bincount(codes,weights=resid); score1=np.bincount(codes,weights=x*resid)
    meat=np.array([[score0@score0,score0@score1],[score0@score1,score1@score1]])
    v=bread@meat@bread; se=float(np.sqrt(max(v[1,1],0))); t=float(beta[1]/se) if se else np.nan
    return float(beta[1]),se,t,_normal_pvalue(t) if np.isfinite(t) else np.nan


def _cluster_scores(x: np.ndarray, residual: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    codes,_=pd.factorize(clusters,sort=False)
    return np.column_stack((np.bincount(codes,weights=residual),np.bincount(codes,weights=x*residual)))


def _two_way_clustered_slope(x: np.ndarray,y: np.ndarray,dates: np.ndarray,symbols: np.ndarray) -> tuple[float,float,float]:
    X=np.column_stack((np.ones(len(x)),x)); beta=np.linalg.pinv(X.T@X)@(X.T@y); variance=_two_way_clustered_covariance(X,y-X@beta,dates,symbols)
    se=float(np.sqrt(max(variance[1,1],0))); t=float(beta[1]/se) if se else np.nan
    return se,t,_normal_pvalue(t) if np.isfinite(t) else np.nan


def _cluster_covariance(X:np.ndarray,residual:np.ndarray,clusters:np.ndarray)->np.ndarray:
    bread=np.linalg.pinv(X.T@X); codes,_=pd.factorize(clusters,sort=False); scores=np.zeros((codes.max()+1,X.shape[1])); np.add.at(scores,codes,X*residual[:,None]); groups=len(scores); n=len(X); k=X.shape[1]
    correction=(groups/(groups-1))*((n-1)/(n-k)) if groups>1 and n>k else 1.0
    return correction*bread@(scores.T@scores)@bread


def _two_way_clustered_covariance(X:np.ndarray,residual:np.ndarray,dates:np.ndarray,symbols:np.ndarray)->np.ndarray:
    intersection=np.char.add(np.char.add(symbols.astype(str),"|"),dates.astype(str))
    return _cluster_covariance(X,residual,dates)+_cluster_covariance(X,residual,symbols)-_cluster_covariance(X,residual,intersection)


def _categorical_clustered_from_codes(category,y,dates,symbols,intersections)->dict:
    category_count=int(category.max())+1 if len(category) else 0
    if category_count<=1:return {"raw_p":np.nan,"two_way_cluster_p":np.nan,"category_count":category_count}
    counts=np.bincount(category,minlength=category_count); sums=np.bincount(category,weights=y,minlength=category_count); means=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    beta=np.r_[means[0],means[1:]-means[0]]; residual=y-means[category]
    xtx=np.zeros((category_count,category_count)); xtx[0,0]=len(y); xtx[0,1:]=counts[1:]; xtx[1:,0]=counts[1:]; xtx[np.arange(1,category_count),np.arange(1,category_count)]=counts[1:]
    bread=np.linalg.pinv(xtx)
    def covariance(codes):
        combined=codes*category_count+category; slots=(int(codes.max())+1)*category_count
        category_scores=np.bincount(combined,weights=residual,minlength=slots).reshape(-1,category_count)
        design_scores=np.column_stack((category_scores.sum(axis=1),category_scores[:,1:]))
        groups=int(np.count_nonzero(np.bincount(codes))); n=len(y); k=category_count
        correction=(groups/(groups-1))*((n-1)/(n-k)) if groups>1 and n>k else 1.0
        return correction*bread@(design_scores.T@design_scores)@bread
    date_cov=covariance(dates); two_cov=date_cov+covariance(symbols)-covariance(intersections)
    def wald(cov):
        b=beta[1:]; v=cov[1:,1:]; statistic=float(b@np.linalg.pinv(v)@b); return float(stats.chi2.sf(statistic,len(b)))
    return {"raw_p":wald(date_cov),"two_way_cluster_p":wald(two_cov),"category_count":category_count}


def _categorical_clustered_test(sub:pd.DataFrame,feature:str,target:str)->dict:
    category,_=pd.factorize(sub[feature],sort=True); dates,_=pd.factorize(sub.session_date,sort=False); symbols,_=pd.factorize(sub.symbol,sort=False)
    intersections=dates*(int(symbols.max())+1)+symbols
    return _categorical_clustered_from_codes(category,sub[target].to_numpy(float),dates,symbols,intersections)


def categorical_scan_batch(frame:pd.DataFrame,features:list[FeatureSpec],targets:list[str],config:ScanConfig)->pd.DataFrame:
    """Exact categorical screen with row-group codes reused across all pairs."""
    eligible=frame.analysis_eligible.fillna(False).to_numpy() if "analysis_eligible" in frame else np.ones(len(frame),dtype=bool)
    base=frame.loc[eligible]
    dates,_=pd.factorize(base.session_date,sort=False); symbols,_=pd.factorize(base.symbol,sort=False); decisions,_=pd.factorize(base.decision_ts,sort=False)
    years,_=pd.factorize(pd.to_datetime(base.session_date).dt.year,sort=False); intersections=dates*(int(symbols.max())+1)+symbols
    sector_codes,_=pd.factorize(base.sector,sort=False) if "sector" in base else (None,None)
    target_values=base[targets].to_numpy(dtype=np.float32,copy=True); rows=[]
    for spec in features:
        category,labels=pd.factorize(base[spec.name],sort=True,use_na_sentinel=True)
        for target_index,target in enumerate(targets):
            y=target_values[:,target_index]; valid=(category>=0)&np.isfinite(y); count=int(valid.sum())
            def groups(codes):
                usable=valid&(codes>=0); return int(np.count_nonzero(np.bincount(codes[usable]))) if usable.any() else 0
            sessions=groups(dates); valid_symbols=groups(symbols); valid_decisions=groups(decisions); valid_years=groups(years); valid_sectors=groups(sector_codes) if sector_codes is not None else 0
            row={"feature":spec.name,"feature_family":spec.family,"feature_classification":spec.classification,"target":target,"table_rows":len(frame),"n":count,"valid_observations":count,"sessions":sessions,"valid_sessions":sessions,"symbols":valid_symbols,"valid_symbols":valid_symbols,"valid_decision_timestamps":valid_decisions,"valid_years":valid_years,"valid_sectors":valid_sectors,"raw_p":np.nan}
            if count<config.min_observations or sessions<config.min_sessions or valid_symbols<config.min_symbols or valid_decisions<config.min_decision_timestamps or valid_years<config.min_years:
                rows.append({**row,"status":"insufficient_data"});continue
            c=category[valid]; values=y[valid]
            clustered=_categorical_clustered_from_codes(c,values,dates[valid],symbols[valid],intersections[valid])
            category_counts=np.bincount(c,minlength=len(labels)); category_sums=np.bincount(c,weights=values,minlength=len(labels)); category_means=np.divide(category_sums,category_counts,out=np.full(len(labels),np.nan),where=category_counts>0)
            rows.append({**row,**clustered,"category_max_minus_min":float(np.nanmax(category_means)-np.nanmin(category_means)),"status":"categorical_screened"})
    return pd.DataFrame(rows)


def binary_scan_batch(frame: pd.DataFrame, features: list[FeatureSpec], targets: list[str], config: ScanConfig) -> pd.DataFrame:
    """Exact binary on/off screen; binary values are never quantile binned."""
    if config.binary_primary_screen_inference != "two_way_date_symbol":
        raise ValueError("Binary screening may not silently fall back from two-way inference")
    eligible=frame.analysis_eligible.fillna(False).to_numpy() if "analysis_eligible" in frame else np.ones(len(frame),dtype=bool)
    base=frame.loc[eligible]; rows=[]
    for spec in features:
        x=pd.to_numeric(base[spec.name],errors="coerce")
        if not x.dropna().isin([0,1]).all():
            raise ValueError(f"Binary feature contains values outside 0/1: {spec.name}")
        for target in targets:
            y=pd.to_numeric(base[target],errors="coerce"); valid=x.notna()&y.notna(); sub=base.loc[valid,["session_date","symbol","decision_ts"]].copy(); sub["x"]=x.loc[valid]; sub["y"]=y.loc[valid]
            count=len(sub); sessions=sub.session_date.nunique(); symbols=sub.symbol.nunique(); decisions=sub.decision_ts.nunique(); years=pd.to_datetime(sub.session_date).dt.year.nunique()
            row={"feature":spec.name,"feature_family":spec.family,"feature_classification":spec.classification,"scan_kind":"binary","discovery_phase":spec.discovery_phase,"arity":spec.arity,"operator":spec.operator,"parent_features":json.dumps(spec.parent_features),"redundancy_group":spec.redundancy_group,"target":target,"target_family":target.removesuffix("_benchmark_adjusted"),"table_rows":len(frame),"n":count,"valid_observations":count,"sessions":sessions,"valid_sessions":sessions,"symbols":symbols,"valid_symbols":symbols,"valid_decision_timestamps":decisions,"valid_years":years,"raw_p":np.nan}
            if count<config.min_observations or sessions<config.min_sessions or symbols<config.min_symbols or decisions<config.min_decision_timestamps or years<config.min_years:
                rows.append({**row,"status":"insufficient_data"}); continue
            on=sub.loc[sub.x.eq(1),"y"]; off=sub.loc[sub.x.eq(0),"y"]
            coverage=binary_coverage(base,spec.name,target); coverage_fields=coverage.as_dict(); coverage_status,coverage_reason=pair_status(coverage,config)
            if not len(on) or not len(off):
                rows.append({**row,"status":"constant_feature"}); continue
            if coverage_status!="sufficient":
                rows.append({**row,**coverage_fields,"coverage_reason":coverage_reason,"status":"insufficient_data"}); continue
            slope,se,t,p=_clustered_slope(sub.x.to_numpy(float),sub.y.to_numpy(float),sub.session_date.astype(str).to_numpy())
            two_way_se,two_way_t,two_way_p=_two_way_clustered_slope(sub.x.to_numpy(float),sub.y.to_numpy(float),sub.session_date.astype(str).to_numpy(),sub.symbol.to_numpy())
            pearson=float(sub.x.corr(sub.y)); spearman=float(sub.x.corr(sub.y,method="spearman"))
            rows.append({**row,**coverage_fields,"on_count":int(len(on)),"off_count":int(len(off)),"on_mean_target":float(on.mean()),"off_mean_target":float(off.mean()),"mean_target":float(sub.y.mean()),"median_target":float(sub.y.median()),"pearson":pearson,"spearman":spearman,"slope":slope,"cluster_se":se,"cluster_t":t,"date_cluster_p":p,"two_way_cluster_se":two_way_se,"two_way_cluster_t":two_way_t,"two_way_cluster_p":two_way_p,"raw_p":two_way_p,"screen_inference":"two_way_date_symbol","top_bottom_spread":slope,"monotonicity":np.nan,"shape":"binary_positive" if slope>0 else "binary_negative","effect_kind":"binary_on_minus_off","effect_value":slope,"status":"binary_screened"})
    return pd.DataFrame(rows)


def _hac_mean(series: pd.Series,max_lag: int|None=None) -> tuple[float,float,float]:
    values=pd.Series(series,dtype=float).dropna().to_numpy()
    if len(values)<3:return (float(np.mean(values)) if len(values) else np.nan,np.nan,np.nan)
    centered=values-values.mean(); lag=max_lag if max_lag is not None else max(1,int(4*(len(values)/100)**(2/9)))
    long_var=float(centered@centered/len(values))
    for k in range(1,min(lag,len(values)-1)+1):
        gamma=float(centered[k:]@centered[:-k]/len(values)); long_var+=2*(1-k/(lag+1))*gamma
    se=float(np.sqrt(max(long_var,0)/len(values))); t=float(values.mean()/se) if se else np.nan
    return float(values.mean()),se,t


def _quantiles(x: pd.Series, y: pd.Series, sessions: pd.Series, q: int, min_bin: int, bootstrap_samples: int=500) -> tuple[dict, pd.DataFrame]:
    try: bins=pd.qcut(x.rank(method="first"),q,labels=False,duplicates="drop")
    except ValueError: return {"top_bottom_spread":np.nan,"monotonicity":np.nan,"shape":"insufficient"},pd.DataFrame()
    work=pd.DataFrame({"bin":bins,"target":y,"session":sessions}).dropna()
    tab=work.groupby("bin",observed=True).target.agg(["count","mean","median",lambda z:(z>0).mean(),"std"]).rename(columns={"<lambda_0>":"win_rate"}).reset_index()
    daily=work.groupby(["session","bin"],observed=True).target.mean().unstack("bin")
    session_stats=[]
    for b in tab.bin:
        mean,se,t=_hac_mean(daily.get(b,pd.Series(dtype=float))); session_stats.append((mean,se,mean-1.96*se if np.isfinite(se) else np.nan,mean+1.96*se if np.isfinite(se) else np.nan))
    tab[["equal_session_mean","se","ci_low","ci_high"]]=pd.DataFrame(session_stats,index=tab.index)
    if len(tab)<3 or tab["count"].min()<min_bin: return {"top_bottom_spread":np.nan,"monotonicity":np.nan,"shape":"insufficient"},tab
    rho=float(stats.spearmanr(tab.bin,tab["mean"]).statistic); spread=float(tab["mean"].iloc[-1]-tab["mean"].iloc[0]); diffs=np.diff(tab["mean"])
    if abs(rho)>=.7: shape="positive_monotonic" if rho>0 else "negative_monotonic"
    elif max(tab["mean"].iloc[0],tab["mean"].iloc[-1])-tab["mean"].iloc[len(tab)//2]>.5*abs(spread): shape="two_sided_tail"
    elif tab["mean"].iloc[len(tab)//2] > max(tab["mean"].iloc[0],tab["mean"].iloc[-1]): shape="inverted_u"
    elif tab["mean"].iloc[len(tab)//2] < min(tab["mean"].iloc[0],tab["mean"].iloc[-1]): shape="u_shaped"
    elif abs(diffs[-1])>2*np.nanmedian(abs(diffs[:-1])): shape="positive_tail" if diffs[-1]>0 else "negative_tail"
    else: shape="no_stable_shape"
    spread_series=(daily.iloc[:,-1]-daily.iloc[:,0]).dropna() if daily.shape[1]>=2 else pd.Series(dtype=float)
    _,spread_se,spread_t=_hac_mean(spread_series)
    if len(spread_series) and bootstrap_samples:
        rng=np.random.default_rng(20260714); boot=np.asarray([rng.choice(spread_series.to_numpy(),len(spread_series),replace=True).mean() for _ in range(bootstrap_samples)])
        boot_low,boot_high=np.quantile(boot,[.025,.975])
    else: boot_low=boot_high=np.nan
    return {"top_bottom_spread":spread,"monotonicity":rho,"shape":shape,"daily_spread_hac_se":spread_se,"daily_spread_hac_t":spread_t,"session_bootstrap_ci_low":boot_low,"session_bootstrap_ci_high":boot_high},tab


def _cross_sectional_ic(frame: pd.DataFrame, feature: str, target: str,min_symbols:int) -> dict:
    groups=frame["decision_ts"]
    xr=frame[feature].groupby(groups,sort=False).rank(method="average")
    yr=frame[target].groupby(groups,sort=False).rank(method="average")
    work=pd.DataFrame({"group":groups,"x":xr,"y":yr}); work["xx"]=xr*xr; work["yy"]=yr*yr; work["xy"]=xr*yr
    agg=work.groupby("group",sort=False).agg(n=("x","size"),sx=("x","sum"),sy=("y","sum"),sxx=("xx","sum"),syy=("yy","sum"),sxy=("xy","sum"))
    cov=agg.sxy-agg.sx*agg.sy/agg.n; vx=agg.sxx-agg.sx*agg.sx/agg.n; vy=agg.syy-agg.sy*agg.sy/agg.n
    ic=(cov/np.sqrt(vx*vy)).where(agg.n.ge(min_symbols)).replace([np.inf,-np.inf],np.nan).dropna()
    if len(ic)<2:return {"ic_mean":np.nan,"ic_median":np.nan,"ic_std":np.nan,"ic_t":np.nan,"ic_positive_pct":np.nan}
    daily=ic.groupby(pd.to_datetime(ic.index,utc=True).date).mean(); mean,se,t=_hac_mean(daily); p=float(2*stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan
    index=pd.to_datetime(ic.index,utc=True); by_year=ic.groupby(index.year).mean().to_dict(); local=index.tz_convert("America/New_York"); by_time=ic.groupby(local.strftime("%H:%M")).mean().to_dict()
    return {"ic_mean":float(ic.mean()),"ic_median":float(ic.median()),"ic_std":float(ic.std(ddof=1)),"ic_information_ratio":float(ic.mean()/ic.std(ddof=1)) if ic.std(ddof=1) else np.nan,"ic_hac_se":se,"ic_hac_t":t,"ic_hac_p":p,"ic_positive_pct":float((ic>0).mean()),"ic_positive_day_pct":float((daily>0).mean()),"ic_timestamps":len(ic),"ic_days":len(daily),"ic_by_year":json.dumps(by_year,sort_keys=True),"ic_by_time_of_day":json.dumps(by_time,sort_keys=True)}


def _outlier_diagnostics(sub:pd.DataFrame,feature:str,target:str,direction:float) -> dict:
    sign=1 if direction>=0 else -1
    def metrics(z):
        if len(z)<10:return (np.nan,np.nan,np.nan,np.nan,np.nan)
        rho=float(z[feature].corr(z[target],method="spearman")); bins=pd.qcut(z[feature].rank(method="first"),10,labels=False,duplicates="drop"); means=z.groupby(bins,observed=True)[target].mean(); spread=float(means.iloc[-1]-means.iloc[0]) if len(means)>1 else np.nan
        _,_,cluster_t,_=_clustered_slope(z[feature].to_numpy(float),z[target].to_numpy(float),z.session_date.astype(str).to_numpy()); daily=z.assign(_bin=bins).groupby(["session_date","_bin"],observed=True)[target].mean().unstack(); daily_spread=(daily.iloc[:,-1]-daily.iloc[:,0])*sign if daily.shape[1]>1 else pd.Series(dtype=float); _,se,_=_hac_mean(daily_spread); signed=spread*sign
        return rho*sign,signed,cluster_t*sign,signed-1.96*se if np.isfinite(se) else np.nan,signed+1.96*se if np.isfinite(se) else np.nan
    _,base_signed,_,_,_=metrics(sub); results={}
    bounds=lambda q:(sub[target].quantile(q),sub[target].quantile(1-q))
    variants={
        "feature_winsor":sub.assign(**{feature:sub[feature].clip(sub[feature].quantile(.01),sub[feature].quantile(.99))}),
        "target_winsor":sub.assign(**{target:sub[target].clip(sub[target].quantile(.01),sub[target].quantile(.99))}),
        "remove_target_001":sub.loc[sub[target].between(*bounds(.001))],"remove_target_005":sub.loc[sub[target].between(*bounds(.005))],"remove_target_01":sub.loc[sub[target].between(*bounds(.01))],
    }
    variants["both_winsor"]=variants["feature_winsor"].assign(**{target:sub[target].clip(sub[target].quantile(.01),sub[target].quantile(.99))})
    day_effect=sub.groupby("session_date").apply(lambda z:_signed_decile_spread(z,feature,target)*sign,include_groups=False).sort_values(ascending=False); symbol_effect=sub.groupby("symbol").apply(lambda z:_signed_decile_spread(z,feature,target)*sign,include_groups=False).sort_values(ascending=False)
    variants["remove_best_day"]=sub.loc[~sub.session_date.isin(day_effect.head(1).index)]; variants["remove_best_five_days"]=sub.loc[~sub.session_date.isin(day_effect.head(5).index)]; variants["remove_best_symbol"]=sub.loc[~sub.symbol.isin(symbol_effect.head(1).index)]
    contribution=(sub[target]*sign).abs(); variants["remove_top_ten_observations"]=sub.drop(contribution.nlargest(min(10,len(sub))).index)
    signed=[]
    for name,z in variants.items():
        rho,spread,t,low,high=metrics(z); results[f"sensitivity_{name}_signed_spearman"]=rho; results[f"sensitivity_{name}_signed_spread"]=spread; results[f"sensitivity_{name}_clustered_t"]=t; results[f"sensitivity_{name}_ci_low"]=low; results[f"sensitivity_{name}_ci_high"]=high
        if np.isfinite(spread):signed.append(spread)
    results["outlier_worst_signed_spread"]=min(signed) if signed else np.nan
    results["outlier_sensitivity"]=max([abs(base_signed-v) for v in signed if np.isfinite(base_signed)] or [np.nan])
    return results


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    numeric=pd.to_numeric(values,errors="coerce"); valid=numeric.notna(); p=numeric[valid].to_numpy()
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):raise ValueError("Benjamini-Hochberg requires finite p-values in [0, 1]")
    order=np.argsort(p); ranked=p[order]; adj=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; out=pd.Series(np.nan,index=values.index); out.loc[numeric[valid].iloc[order].index]=np.minimum(adj,1); return out


def _signed_decile_spread(frame:pd.DataFrame,feature:str,target:str)->float:
    if len(frame)<20:return np.nan
    values=pd.to_numeric(frame[feature],errors="coerce")
    finite=values.dropna()
    if len(finite) and finite.isin([0,1]).all():
        on=frame.loc[values.eq(1),target]; off=frame.loc[values.eq(0),target]
        return float(on.mean()-off.mean()) if len(on) and len(off) else np.nan
    try:bins=pd.qcut(frame[feature].rank(method="first"),10,labels=False,duplicates="drop")
    except ValueError:return np.nan
    means=frame.groupby(bins,observed=True)[target].mean()
    return float(means.iloc[-1]-means.iloc[0]) if len(means)>1 else np.nan


def scan(frame: pd.DataFrame, features: list[FeatureSpec], targets: list[str], config: ScanConfig, checkpoint_path=None, *, skip_dense: bool=False, direction_hint: float | None=None) -> tuple[pd.DataFrame, dict[tuple[str,str],pd.DataFrame]]:
    rows=[]; quantile_tables={}; backend=CorrelationBackend(config.use_cuda,config.cuda_device)
    completed=set()
    if checkpoint_path is not None and config.resume and pd.io.common.file_exists(checkpoint_path):
        prior=pd.read_csv(checkpoint_path); rows=prior.to_dict("records"); completed={(r["feature"],r["target"]) for r in rows}
    for spec in features:
        for target in targets:
            if (spec.name,target) in completed: continue
            eligible=frame.analysis_eligible.fillna(False) if "analysis_eligible" in frame else pd.Series(True,index=frame.index)
            columns=["session_date","symbol","decision_ts",spec.name,target]+(["sector"] if "sector" in frame else [])
            sub=frame.loc[eligible,columns].replace([np.inf,-np.inf],np.nan).dropna(subset=[spec.name,target])
            base={"feature":spec.name,"feature_family":spec.family,"feature_classification":spec.classification,"target":target,"table_rows":len(frame),"n":len(sub),"valid_observations":len(sub),"sessions":sub.session_date.nunique(),"valid_sessions":sub.session_date.nunique(),"symbols":sub.symbol.nunique(),"valid_symbols":sub.symbol.nunique(),"valid_decision_timestamps":sub.decision_ts.nunique(),"valid_years":pd.to_datetime(sub.session_date).dt.year.nunique(),"valid_sectors":sub.sector.nunique() if "sector" in sub else 0,"raw_p":np.nan}
            if len(sub)<config.min_observations or base["sessions"]<config.min_sessions or base["symbols"]<config.min_symbols or base["valid_decision_timestamps"]<config.min_decision_timestamps or base["valid_years"]<config.min_years:
                rows.append({**base,"status":"insufficient_data"});continue
            x=sub[spec.name].astype(float);y=sub[target].astype(float)
            if spec.classification=="categorical" or spec.dtype=="categorical":
                categories=sub.groupby(spec.name)[target].agg(["count","mean","median"]); groups=[z[target].to_numpy() for _,z in sub.groupby(spec.name) if len(z)>=config.min_bin_observations]
                clustered=_categorical_clustered_test(sub,spec.name,target)
                rows.append({**base,**clustered,"category_max_minus_min":float(categories["mean"].max()-categories["mean"].min()),"status":"categorical_screened"})
                quantile_tables[(spec.name,target)]=categories.reset_index().rename(columns={spec.name:"category"})
                continue
            if feature_scan_kind(spec)=="binary":
                if not x.isin([0,1]).all():
                    raise ValueError(f"Binary feature contains values outside 0/1: {spec.name}")
                coverage=binary_coverage(sub,spec.name,target);coverage_fields=coverage.as_dict();coverage_status,coverage_reason=pair_status(coverage,config)
                on=y.loc[x.eq(1)]; off=y.loc[x.eq(0)]
                if not len(on) or not len(off):
                    rows.append({**base,**coverage_fields,"status":"constant_feature"}); continue
                if coverage_status!="sufficient":
                    rows.append({**base,**coverage_fields,"coverage_reason":coverage_reason,"status":"insufficient_data"});continue
                slope,se,t,p=_clustered_slope(x.to_numpy(),y.to_numpy(),sub.session_date.astype(str).to_numpy())
                two_way_se,two_way_t,two_way_p=_two_way_clustered_slope(x.to_numpy(),y.to_numpy(),sub.session_date.astype(str).to_numpy(),sub.symbol.to_numpy())
                effect=float(on.mean()-off.mean()); direction=np.sign(direction_hint or effect or 1)
                annual=sub.assign(year=pd.to_datetime(sub.session_date).dt.year).groupby("year").apply(lambda z: z.loc[z[spec.name].eq(1),target].mean()-z.loc[z[spec.name].eq(0),target].mean(),include_groups=False).dropna()
                rows.append({**base,**coverage_fields,"scan_kind":"binary","on_count":len(on),"off_count":len(off),"on_mean_target":float(on.mean()),"off_mean_target":float(off.mean()),"mean_target":float(y.mean()),"median_target":float(y.median()),"std_target":float(y.std()),"win_rate":float((y>0).mean()),"pearson":float(x.corr(y)),"spearman":float(x.corr(y,method="spearman")),"slope":slope,"cluster_se":se,"cluster_t":t,"date_cluster_p":p,"two_way_cluster_se":two_way_se,"two_way_cluster_t":two_way_t,"two_way_cluster_p":two_way_p,"raw_p":two_way_p,"screen_inference":"two_way_date_symbol","ci_low":slope-1.96*two_way_se,"ci_high":slope+1.96*two_way_se,"top_bottom_spread":effect,"monotonicity":np.nan,"shape":"binary_positive" if effect>0 else "binary_negative","effect_kind":"binary_on_minus_off","effect_value":effect,"year_consistency":float((annual*direction>0).mean()) if len(annual) else np.nan,"symbol_breadth":np.nan,"time_stability":np.nan,"outlier_worst_signed_spread":np.nan,"outlier_sensitivity":np.nan,"status":"binary_screened"})
                quantile_tables[(spec.name,target)]=pd.DataFrame([{"signal":0,"count":len(off),"mean":off.mean(),"median":off.median()},{"signal":1,"count":len(on),"mean":on.mean(),"median":on.median()}])
                continue
            slope,se,t,p=_clustered_slope(x.to_numpy(),y.to_numpy(),sub.session_date.astype(str).to_numpy())
            two_way_se,two_way_t,two_way_p=_two_way_clustered_slope(x.to_numpy(),y.to_numpy(),sub.session_date.astype(str).to_numpy(),sub.symbol.to_numpy())
            if skip_dense:
                qstats={"top_bottom_spread":np.nan,"monotonicity":np.nan,"shape":"pending_gpu"}; qtab=pd.DataFrame(); pearson=spearman=np.nan
            else:
                qstats,qtab=_quantiles(x,y,sub.session_date,config.quantiles,config.min_bin_observations,config.exact_bootstrap_samples); pearson,spearman=backend.correlations(x.to_numpy(),y.to_numpy())
            annual_effect=sub.assign(year=pd.to_datetime(sub.session_date).dt.year).groupby("year").apply(lambda z:_signed_decile_spread(z,spec.name,target),include_groups=False).dropna()
            symbol_effect=sub.groupby("symbol").filter(lambda z:len(z)>=50).groupby("symbol").apply(lambda z:_signed_decile_spread(z,spec.name,target),include_groups=False).dropna()
            cs=_cross_sectional_ic(sub,spec.name,target,config.cross_sectional_min_symbols) if spec.classification=="cross_sectional" else {}
            status="statistically_interesting" if p<.05 else "no_meaningful_relationship"
            local=pd.to_datetime(sub.decision_ts,utc=True).dt.tz_convert("America/New_York"); sub=sub.assign(time_bucket=(local.dt.hour*60+local.dt.minute).floordiv(60),year=pd.to_datetime(sub.session_date).dt.year)
            time_corr=sub.groupby("time_bucket").apply(lambda z:z[spec.name].corr(z[target],method="spearman"),include_groups=False).dropna()
            direction=spearman if np.isfinite(spearman) else direction_hint; signed_direction=np.sign(direction) if direction is not None and np.isfinite(direction) else 1
            robustness={} if skip_dense else _outlier_diagnostics(sub,spec.name,target,direction)
            rows.append({**base,"mean_target":float(y.mean()),"median_target":float(y.median()),"std_target":float(y.std()),"win_rate":float((y>0).mean()),"skewness":float(stats.skew(y,nan_policy="omit")),"downside_p05":float(y.quantile(.05)),"upside_p95":float(y.quantile(.95)),"pearson":pearson,"spearman":spearman,"slope":slope,"cluster_se":se,"cluster_t":t,"two_way_cluster_se":two_way_se,"two_way_cluster_t":two_way_t,"two_way_cluster_p":two_way_p,"raw_p":p,"ci_low":slope-1.96*se,"ci_high":slope+1.96*se,"year_consistency":float((annual_effect*signed_direction>0).mean()) if len(annual_effect) else np.nan,"symbol_breadth":float((symbol_effect*signed_direction>0).mean()) if len(symbol_effect) else np.nan,"median_annual_effect":float((annual_effect*signed_direction).median()) if len(annual_effect) else np.nan,"worst_annual_effect":float((annual_effect*signed_direction).min()) if len(annual_effect) else np.nan,"lower_quartile_annual_effect":float((annual_effect*signed_direction).quantile(.25)) if len(annual_effect) else np.nan,"years_above_minimum_effect":float((annual_effect*signed_direction>=config.minimum_effect_bps/10000).mean()) if len(annual_effect) else np.nan,"leave_one_year_out_effect":float((annual_effect*signed_direction).drop(annual_effect.idxmax()).mean()) if len(annual_effect)>1 else np.nan,"leave_one_symbol_out_effect":float((symbol_effect*signed_direction).drop(symbol_effect.idxmax()).mean()) if len(symbol_effect)>1 else np.nan,"time_stability":float((time_corr*signed_direction>0).mean()) if len(time_corr) else np.nan,"status":status,**qstats,**cs,**robustness})
            quantile_tables[(spec.name,target)]=qtab
            if checkpoint_path is not None and len(rows)%config.checkpoint_every_pairs==0: pd.DataFrame(rows).to_csv(checkpoint_path,index=False)
    result=pd.DataFrame(rows)
    if not result.empty:
        for column in ["top_bottom_spread","monotonicity","year_consistency","symbol_breadth","outlier_sensitivity","cluster_t"]:
            if column not in result:result[column]=np.nan
        result["target_family"]=result.target.str.replace(r"_benchmark_adjusted$","",regex=True)
        result["bh_fdr_p"]=result.groupby(["feature_family","target_family"],dropna=False).raw_p.transform(benjamini_hochberg)
        result["test_count"]=int(result.raw_p.notna().sum())
        result["redundancy_group"]=result.feature.str.replace(r"_(1|2|3|5|10|15|30|60|1560|4680)$","_LOOKBACK",regex=True)
        effect=result.top_bottom_spread.abs().fillna(0).rank(pct=True); significance=(1-result.bh_fdr_p.fillna(1)).clip(0,1); stability=result[["year_consistency","symbol_breadth"]].mean(axis=1).fillna(0)
        result["anomaly_score"]=.35*effect+.35*significance+.2*result.monotonicity.abs().fillna(0)+.1*stability-.2*result.outlier_sensitivity.fillna(0)
        result.loc[(result.bh_fdr_p<.05)&(result.year_consistency>=.6)&(result.symbol_breadth>=.6),"status"]="robust_anomaly_candidate"
        result=result.sort_values("anomaly_score",ascending=False,na_position="last")
    if checkpoint_path is not None: result.to_csv(checkpoint_path,index=False)
    return result,quantile_tables
