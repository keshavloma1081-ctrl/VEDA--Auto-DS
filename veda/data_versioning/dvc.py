import os, hashlib, json, logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
REGISTRY_PATH = os.getenv("DATA_REGISTRY_PATH", "data/registry.json")

@dataclass
class DatasetVersion:
    version_id: str
    file_path: str
    file_name: str
    file_hash: str
    file_size_bytes: int
    num_rows: int
    num_cols: int
    num_missing: int
    missing_pct: float
    column_names: List[str]
    dtypes: Dict[str, str]
    statistics: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    workflow_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parent_version_id: Optional[str] = None

    def to_dict(self):
        return {
            "version_id": self.version_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes/(1024*1024), 3),
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "num_missing": self.num_missing,
            "missing_pct": round(self.missing_pct, 2),
            "column_names": self.column_names,
            "dtypes": self.dtypes,
            "statistics": self.statistics,
            "tags": self.tags,
            "notes": self.notes,
            "workflow_id": self.workflow_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "parent_version_id": self.parent_version_id
        }

class DatasetProfiler:
    @staticmethod
    def compute_file_hash(file_path):
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def profile_dataframe(df):
        stats = {}
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(float(df[col].isnull().mean()*100), 2),
                "unique_count": int(df[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                col_stats.update({
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                    "p25": round(float(df[col].quantile(0.25)), 4),
                    "p50": round(float(df[col].quantile(0.50)), 4),
                    "p75": round(float(df[col].quantile(0.75)), 4),
                })
            else:
                top_vals = df[col].value_counts().head(5).to_dict()
                col_stats["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
            stats[col] = col_stats
        return stats

    @classmethod
    def profile_file(cls, file_path):
        path = Path(file_path)
        file_size = path.stat().st_size
        file_hash = cls.compute_file_hash(file_path)
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xlsx",".xls"]:
            df = pd.read_excel(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported: {ext}")
        return {
            "file_hash": file_hash,
            "file_size_bytes": file_size,
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "num_missing": int(df.isnull().sum().sum()),
            "missing_pct": float(df.isnull().mean().mean()*100),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "statistics": cls.profile_dataframe(df)
        }

class DataVersionControl:
    def __init__(self, registry_path=REGISTRY_PATH):
        self.registry_path = registry_path
        self._registry: Dict[str, DatasetVersion] = {}
        self._hash_index: Dict[str, str] = {}
        self._load_registry()

    def _load_registry(self):
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                for vid, vdata in data.items():
                    self._registry[vid] = DatasetVersion(**vdata)
                    self._hash_index[vdata["file_hash"]] = vid
        except Exception as e:
            log.warning(f"Could not load registry: {e}")

    def _save_registry(self):
        try:
            os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
            data = {vid: v.to_dict() for vid, v in self._registry.items()}
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.warning(f"Could not save registry: {e}")

    def register(self, file_path, tags=None, notes="", workflow_id=None, created_by=None, parent_version_id=None):
        tags = tags or []
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        profile = DatasetProfiler.profile_file(file_path)
        existing_id = self._hash_index.get(profile["file_hash"])
        if existing_id and existing_id in self._registry:
            log.info(f"Dataset already registered as {existing_id}")
            return self._registry[existing_id]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_hash = profile["file_hash"][:8]
        version_id = f"v_{ts}_{short_hash}"
        version = DatasetVersion(
            version_id=version_id,
            file_path=str(path.resolve()),
            file_name=path.name,
            tags=tags, notes=notes,
            workflow_id=workflow_id,
            created_by=created_by,
            parent_version_id=parent_version_id,
            **profile
        )
        self._registry[version_id] = version
        self._hash_index[profile["file_hash"]] = version_id
        self._save_registry()
        self._save_to_db(version)
        log.info(f"Dataset registered | id={version_id} | rows={profile['num_rows']}")
        return version

    def get(self, version_id):
        return self._registry.get(version_id)

    def get_by_hash(self, file_hash):
        vid = self._hash_index.get(file_hash)
        return self._registry.get(vid) if vid else None

    def list_versions(self, tag=None, workflow_id=None, limit=20):
        versions = list(self._registry.values())
        if tag:
            versions = [v for v in versions if tag in v.tags]
        if workflow_id:
            versions = [v for v in versions if v.workflow_id == workflow_id]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)[:limit]

    def get_lineage(self, version_id):
        lineage = []
        current_id = version_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            version = self._registry.get(current_id)
            if not version:
                break
            lineage.append({
                "version_id": version.version_id,
                "file_name": version.file_name,
                "created_at": version.created_at,
                "num_rows": version.num_rows,
                "num_cols": version.num_cols,
                "tags": version.tags
            })
            current_id = version.parent_version_id
        return lineage

    def compare(self, version_id_a, version_id_b):
        va = self._registry.get(version_id_a)
        vb = self._registry.get(version_id_b)
        if not va or not vb:
            return {"error": "One or both versions not found"}
        shared = set(va.column_names) & set(vb.column_names)
        only_a = set(va.column_names) - set(vb.column_names)
        only_b = set(vb.column_names) - set(va.column_names)
        return {
            "version_a": version_id_a,
            "version_b": version_id_b,
            "rows": {"a": va.num_rows, "b": vb.num_rows, "diff": vb.num_rows - va.num_rows},
            "cols": {"a": va.num_cols, "b": vb.num_cols, "diff": vb.num_cols - va.num_cols},
            "missing_pct": {"a": va.missing_pct, "b": vb.missing_pct},
            "shared_columns": len(shared),
            "only_in_a": list(only_a),
            "only_in_b": list(only_b),
            "same_hash": va.file_hash == vb.file_hash
        }

    def tag(self, version_id, tags):
        version = self._registry.get(version_id)
        if not version:
            return False
        version.tags = list(set(version.tags + tags))
        self._save_registry()
        return True

    def get_stats(self):
        versions = list(self._registry.values())
        total_size = sum(v.file_size_bytes for v in versions)
        return {
            "total_versions": len(versions),
            "total_size_mb": round(total_size/(1024*1024), 2),
            "unique_files": len(set(v.file_name for v in versions)),
            "tagged_versions": len([v for v in versions if v.tags])
        }

    def _save_to_db(self, version):
        try:
            from veda.database.models import SessionLocal, DatasetVersion as DBDatasetVersion
            db = SessionLocal()
            try:
                record = DBDatasetVersion(
                    version_id=version.version_id,
                    file_path=version.file_path,
                    file_name=version.file_name,
                    file_hash=version.file_hash,
                    file_size_bytes=version.file_size_bytes,
                    num_rows=version.num_rows,
                    num_cols=version.num_cols,
                    num_missing=version.num_missing,
                    column_names=version.column_names,
                    dtypes=version.dtypes,
                    tags=version.tags,
                    notes=version.notes,
                    workflow_id=version.workflow_id,
                    created_by=version.created_by
                )
                db.add(record)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.warning(f"Could not save to DB: {e}")

dvc = DataVersionControl()
