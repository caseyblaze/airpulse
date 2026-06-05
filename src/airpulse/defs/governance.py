"""Data governance primitives: classification levels, ownership, and a
reusable masking helper.

This dataset (Taiwan EPA open air-quality data) carries no PII, so everything
here is classified PUBLIC. The classification + masking framework is included
to demonstrate how the same pipeline would handle CONFIDENTIAL / RESTRICTED
data — e.g. patient records (health) or account-level financial data — by
codifying classification as asset tags and providing a single audited masking
path for sensitive fields.
"""

import hashlib

# --- Classification levels (rises in sensitivity) ---------------------------
PUBLIC = "public"  # open data, no restrictions
INTERNAL = "internal"  # business-internal, non-sensitive
CONFIDENTIAL = "confidential"  # commercially sensitive
RESTRICTED = "restricted"  # PII / regulated (PDPA, HIPAA-like, financial)

# Standard ownership + classification tags applied across the pipeline so the
# Dagster catalog shows governance metadata on every asset.
DATA_OWNER = "team:data-engineering"

PUBLIC_TAGS = {
    "data_classification": PUBLIC,
    "domain": "air-quality",
    "source": "taiwan-epa-moenv",
}


def mask_pii(value: str | None, *, keep_prefix: int = 0) -> str | None:
    """Deterministically pseudonymize a sensitive value via salted-style hashing.

    Deterministic so the same input always maps to the same token (joins still
    work) while the original is irrecoverable. Unused here (no PII in this
    dataset) but kept as the single, auditable masking path a RESTRICTED field
    would flow through before storage.
    """
    if value is None:
        return None
    prefix = value[:keep_prefix]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}***{digest}"
