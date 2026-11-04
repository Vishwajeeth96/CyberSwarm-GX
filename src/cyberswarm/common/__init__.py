"""Shared, dependency-light utilities for the CyberSwarm foundation (stage S1).

This package depends only on :mod:`cyberswarm.contracts`, NumPy and the Python
standard library; ``torch`` is imported lazily inside :mod:`.seeding` only. It
provides the four cross-cutting concerns used by every later stage:

* :mod:`.seeding` - deterministic seed derivation and global seeding.
* :mod:`.hashing` - canonical JSON and SHA-256 digests over records and files.
* :mod:`.logging` - JSON-line ``LogRecord`` emission and the single wall-clock read.
* :mod:`.config` - typed TOML loading that rejects unknown keys, plus a
  canonical resolved-configuration snapshot.
"""

from __future__ import annotations

from .config import (
    ConfigModel,
    config_digest,
    load_toml,
    loads_toml,
    snapshot_config,
    validate_config,
)
from .hashing import (
    canonical_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from .logging import RunLogger, get_logger, utc_now_iso
from .seeding import SeedStreams, derive_seeds, seed_everything

__all__ = [
    # seeding
    "SeedStreams",
    "derive_seeds",
    "seed_everything",
    # hashing
    "canonical_json",
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    # logging
    "RunLogger",
    "get_logger",
    "utc_now_iso",
    # config
    "ConfigModel",
    "config_digest",
    "load_toml",
    "loads_toml",
    "snapshot_config",
    "validate_config",
]
