from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import exchange_calendars

from .config import ScanConfig
from .registry import FeatureSpec,TargetSpec
from .table import source_provenance


def file_hash(path:str|Path|None)->str|None:
    if not path:return None
    source=Path(path)
    if not source.exists():return None
    digest=hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def run_fingerprint(config:ScanConfig,features:list[FeatureSpec],targets:list[TargetSpec],git_revision:str|None,extra_components:dict|None=None)->dict:
    provenance=source_provenance(config)
    package_root=Path(__file__).parent
    source_digest=hashlib.sha256()
    for source in sorted(package_root.glob("*.py")):
        source_digest.update(source.name.encode()); source_digest.update(source.read_bytes())
    components={
        "cache_schema_version":config.cache_schema_version,
        "configuration":config.as_dict(),
        "git_revision":git_revision,
        "pipeline_source_hash":source_digest.hexdigest(),
        "source":provenance,
        "feature_registry": [asdict(x) for x in features],
        "target_registry": [asdict(x) for x in targets],
        "membership_source":config.membership_table,
        "corporate_actions_hash":file_hash(config.corporate_actions_path),
        "sector_map_hash":file_hash(config.sector_map_path),
        "calendar_package_version":exchange_calendars.__version__,
        "calendar_name":config.exchange_calendar,
    }
    if extra_components:
        components["extra_components"] = extra_components
    encoded=json.dumps(components,sort_keys=True,default=str,separators=(",",":")).encode()
    return {"sha256":hashlib.sha256(encoded).hexdigest(),"components":components}


def phase1b_run_fingerprint(config:ScanConfig,features:list[FeatureSpec],targets:list[TargetSpec],extra_components:dict)->dict:
    """Fingerprint a derived run without querying or reading raw source data."""
    package_root=Path(__file__).parent
    source_digest=hashlib.sha256()
    for source in sorted(package_root.glob("*.py")):
        source_digest.update(source.name.encode());source_digest.update(source.read_bytes())
    components={
        "run_type":"phase1b_derived",
        "configuration":config.as_dict(),
        "scan_schema_version":config.scan_schema_version,
        "dual_cache_schema_version":config.dual_cache_schema_version,
        "pipeline_source_hash":source_digest.hexdigest(),
        "feature_registry":[asdict(item) for item in features],
        "target_registry":[asdict(item) for item in targets],
        "extra_components":extra_components,
    }
    encoded=json.dumps(components,sort_keys=True,default=str,separators=(",",":")).encode()
    return {"sha256":hashlib.sha256(encoded).hexdigest(),"components":components}


def enforce_fingerprint(root:Path,fingerprint:dict,resume:bool)->None:
    root.mkdir(parents=True,exist_ok=True)
    path=root/"fingerprint.json"
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf-8"))
        if previous.get("sha256")!=fingerprint["sha256"]:
            raise RuntimeError("Run fingerprint changed; refusing to reuse incompatible caches. Use a new experiment directory or remove the old run directory.")
        if not resume:raise RuntimeError("Run directory already exists and resume is disabled")
    else:
        temporary=path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(fingerprint,indent=2,default=str),encoding="utf-8")
        temporary.replace(path)
