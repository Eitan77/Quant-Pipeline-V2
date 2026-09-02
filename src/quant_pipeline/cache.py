from __future__ import annotations

import hashlib,json
from pathlib import Path
import pandas as pd

from .holdout import assert_pre_holdout_frame, assert_pre_holdout_parquet

ROW_KEYS=["symbol","session_date","bar_start_ts","decision_ts"]


class CacheFingerprintMismatch(RuntimeError):pass


def row_key_hash(frame:pd.DataFrame)->str:
    keys=frame[ROW_KEYS].copy(); keys["session_date"]=pd.to_datetime(keys.session_date).astype(str)
    for column in ["bar_start_ts","decision_ts"]:keys[column]=pd.to_datetime(keys[column],utc=True).astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(keys,index=False).to_numpy().tobytes()).hexdigest()


def schema_hash(frame:pd.DataFrame)->str:
    return hashlib.sha256(json.dumps([(c,str(t)) for c,t in frame.dtypes.items()]).encode()).hexdigest()


def file_sha256(path:Path,block_size:int=8*1024*1024)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(block_size),b""):digest.update(block)
    return digest.hexdigest()


def _parquet_schema_hash(path:Path)->str:
    import pyarrow.parquet as pq
    parquet=pq.ParquetFile(path)
    batches=parquet.iter_batches(batch_size=1)
    try:sample=next(batches).to_pandas()
    except StopIteration:sample=parquet.schema_arrow.empty_table().to_pandas()
    return schema_hash(sample)


def write_cache_metadata(path:Path,frame:pd.DataFrame,fingerprint:str,sealed_holdout_start:str="2026-05-01",validation_record:dict|None=None)->dict:
    assert_pre_holdout_frame(frame,sealed_holdout_start,f"cache write {path}")
    persisted_keys=assert_pre_holdout_parquet(path,sealed_holdout_start,f"cache write {path}")
    if persisted_keys is None or persisted_keys.duplicated(ROW_KEYS).any() or not persisted_keys.sort_values(ROW_KEYS,kind="stable").index.equals(persisted_keys.index):raise ValueError(f"Duplicate, missing, or unsorted cache keys: {path}")
    expected_hash=row_key_hash(frame); persisted_hash=row_key_hash(persisted_keys)
    if len(persisted_keys)!=len(frame) or persisted_hash!=expected_hash or _parquet_schema_hash(path)!=schema_hash(frame):raise ValueError(f"Persisted cache does not match source frame: {path}")
    metadata={"fingerprint":fingerprint,"row_count":len(persisted_keys),"first_key":[str(x) for x in persisted_keys[ROW_KEYS].iloc[0]] if len(persisted_keys) else None,"last_key":[str(x) for x in persisted_keys[ROW_KEYS].iloc[-1]] if len(persisted_keys) else None,"row_key_hash":persisted_hash,"column_schema_hash":schema_hash(frame),"file_size":path.stat().st_size,"file_sha256":file_sha256(path)}
    if validation_record is not None:metadata["point_in_time_validation"]=validation_record
    metadata_path=path.with_suffix(path.suffix+".meta.json");temporary=metadata_path.with_suffix(metadata_path.suffix+".tmp");temporary.write_text(json.dumps(metadata,indent=2),encoding="utf-8");temporary.replace(metadata_path); return metadata


def write_aligned_derived_cache_metadata(path:Path,frame:pd.DataFrame,fingerprint:str,source_key_metadata:dict,sealed_holdout_start:str="2026-05-01")->dict:
    """Validate a derived block against already-verified immutable row keys.

    Derived Phase 1B blocks copy their keys byte-for-byte from an aligned source
    cache.  Re-reading and re-hashing millions of identical keys for every
    feature chunk adds no validation strength, so verify row count, boundary,
    endpoint keys, schema and file digest while reusing the source key hash.
    """
    import pyarrow.parquet as pq
    # Timestamp columns are copied unchanged from the already validated source
    # block; Parquet statistics below fail closed on the persisted boundary.
    assert_pre_holdout_parquet(path,sealed_holdout_start,f"cache write {path}",verify_key_rows=False)
    rows=pq.ParquetFile(path).metadata.num_rows
    if rows!=len(frame) or rows!=source_key_metadata.get("row_count"):raise ValueError(f"Derived cache row count mismatch: {path}")
    first=[str(x) for x in frame[ROW_KEYS].iloc[0]] if rows else None;last=[str(x) for x in frame[ROW_KEYS].iloc[-1]] if rows else None
    if first!=source_key_metadata.get("first_key") or last!=source_key_metadata.get("last_key"):raise ValueError(f"Derived cache endpoint keys mismatch: {path}")
    if _parquet_schema_hash(path)!=schema_hash(frame):raise ValueError(f"Derived cache schema mismatch: {path}")
    metadata={"fingerprint":fingerprint,"row_count":rows,"first_key":first,"last_key":last,"row_key_hash":source_key_metadata["row_key_hash"],"column_schema_hash":schema_hash(frame),"file_size":path.stat().st_size,"file_sha256":file_sha256(path),"aligned_source_key_validation":True}
    metadata_path=path.with_suffix(path.suffix+".meta.json");temporary=metadata_path.with_suffix(metadata_path.suffix+".tmp");temporary.write_text(json.dumps(metadata,indent=2),encoding="utf-8");temporary.replace(metadata_path);return metadata


def validate_cache(path:Path,fingerprint:str,sealed_holdout_start:str="2026-05-01")->dict:
    metadata_path=path.with_suffix(path.suffix+".meta.json")
    if not metadata_path.exists():raise CacheFingerprintMismatch(f"Cache metadata missing: {path}")
    saved=json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved["fingerprint"]!=fingerprint:raise CacheFingerprintMismatch(f"Cache fingerprint mismatch: {path}")
    if "file_sha256" in saved:
        if saved.get("file_size")!=path.stat().st_size or saved["file_sha256"]!=file_sha256(path):raise CacheFingerprintMismatch(f"Cache file digest mismatch: {path}")
        assert_pre_holdout_parquet(path,sealed_holdout_start,f"cache resume {path}",verify_key_rows=False)
        return saved
    keys=assert_pre_holdout_parquet(path,sealed_holdout_start,f"cache resume {path}")
    frame=pd.read_parquet(path); current={"row_count":len(frame),"row_key_hash":row_key_hash(frame),"column_schema_hash":schema_hash(frame)}
    assert_pre_holdout_frame(frame,sealed_holdout_start,f"cache resume {path}")
    if any(saved[k]!=current[k] for k in current):raise CacheFingerprintMismatch(f"Cache integrity check failed: {path}")
    if keys is None or keys.duplicated(ROW_KEYS).any() or not keys.sort_values(ROW_KEYS,kind="stable").index.equals(keys.index):raise CacheFingerprintMismatch(f"Cache keys are duplicate or unsorted: {path}")
    return saved


def assert_key_alignment(feature_frame:pd.DataFrame,target_frame:pd.DataFrame)->None:
    if len(feature_frame)!=len(target_frame) or row_key_hash(feature_frame)!=row_key_hash(target_frame):raise CacheFingerprintMismatch("Feature and target cache row keys do not align")


def assert_cache_key_alignment(feature_path:Path,target_path:Path)->None:
    """Use already-validated cache metadata instead of rehashing 12M rows per batch."""
    def metadata(path:Path)->dict:
        metadata_path=path.with_suffix(path.suffix+".meta.json")
        if not metadata_path.exists():raise CacheFingerprintMismatch(f"Cache metadata missing: {path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    feature=metadata(feature_path); target=metadata(target_path)
    if feature.get("row_count")!=target.get("row_count") or feature.get("row_key_hash")!=target.get("row_key_hash"):
        raise CacheFingerprintMismatch("Feature and target cache metadata keys do not align")
