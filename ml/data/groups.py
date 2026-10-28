"""Printed-card group key, shared by the split and the card-grouped CV.

Certs of the same printed card share a key, so `split.py` can keep a card in
one split and `ml.training.probe` can keep it in one CV fold. Subject variants
(`CHARIZARD-HOLO` / `CHARIZARD-REV.FOIL`) are not normalised: they are
separate keys.
"""

from __future__ import annotations

import hashlib

SPLIT_SALT = "pokescan-split-v1"


def card_group_key(brand: str | None, subject: str | None, number: str | None) -> str:
    return f"{brand or ''}|{subject or ''}|{number or ''}"


def group_bucket(key: str, salt: str = SPLIT_SALT) -> float:
    """Deterministic position of a group in [0, 1): the same key always lands
    in the same bucket, whatever other rows exist."""
    digest = hashlib.sha1(f"{salt}|{key}".encode()).hexdigest()
    return int(digest[:8], 16) / 2**32
