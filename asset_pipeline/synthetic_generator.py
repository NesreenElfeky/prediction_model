from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import random

import pandas as pd


ASSET_TYPES = ["Application", "Server", "Database", "Network Device", "Cloud Service"]
SERVICES = ["http", "https", "ssh", "rdp", "postgres", "mysql", "smtp"]
PORT_BY_SERVICE = {"http": 80, "https": 443, "ssh": 22, "rdp": 3389, "postgres": 5432, "mysql": 3306, "smtp": 25}


def _asset_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12].upper()
    return f"ASSET-{digest}"


def generate_synthetic_assets(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for idx in range(n):
        service_sample = rng.sample(SERVICES, k=rng.randint(1, 4))
        asset_type = rng.choice(ASSET_TYPES)
        domain = f"asset-{idx:04d}.example.internal"
        rows.append(
            {
                "asset_id": _asset_id(domain),
                "asset_name": domain.split(".")[0],
                "asset_type": asset_type,
                "host": f"10.{rng.randint(0, 20)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                "domain": domain,
                "ports": [PORT_BY_SERVICE[s] for s in service_sample],
                "services": [{"port": PORT_BY_SERVICE[s], "service": s, "version": ""} for s in service_sample],
                "tags": [asset_type.lower().replace(" ", "_")],
                "discovered_by": ["synthetic"],
                "internet_exposed": any(s in {"http", "https", "ssh"} for s in service_sample),
                "business_criticality": rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                "last_seen": now,
            }
        )
    return pd.DataFrame(rows)

