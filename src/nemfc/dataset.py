"""Loading the feature mart and labelling its episodes.

Shared by the baseline and training scripts so the two cannot drift into
scoring subtly different row sets, which would make every comparison
between them meaningless without anything failing.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from nemfc.evaluate import assign_episodes, find_episodes

FEATURES_SQL = """
select *
from staging.mart_spike_features
where region_id = :region
order by prediction_at
"""
"""Every column, not a hand-picked list.

The first version named the nine columns Day 5 happened to need, and Day 6 asked
the model for thirty-seven. The backtest died on the first fold. Nothing in the
suite caught it, because every synthetic fixture fabricates whatever columns are
missing -- so the one thing never tested was whether the query returns what the
feature list asks for.

Selecting everything removes the class of failure rather than patching this
instance of it: a feature added to the mart is available without a second edit
here, and `nemfc.model.FORBIDDEN_FEATURES` is what keeps target-derived columns
out of the model. `tests/test_leakage_against_raw.py` now also checks the mart's
real columns against the feature list, so a column disappearing from the
warehouse fails loudly instead of at fold zero.
"""

HISTORY_SQL = """
select settlement_end, spacing_minutes, is_spike
from staging.stg_price_history
where region_id = :region
"""


def load(engine, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_sql(text(FEATURES_SQL), engine, params={"region": region})
    features["prediction_at"] = pd.to_datetime(features["prediction_at"], utc=True)
    features["anchor_settlement_end"] = pd.to_datetime(features["anchor_settlement_end"], utc=True)
    features["is_spike"] = features["is_spike"].astype(bool)

    history = pd.read_sql(text(HISTORY_SQL), engine, params={"region": region})
    history["settlement_end"] = pd.to_datetime(history["settlement_end"], utc=True)
    return features, history


def add_episodes(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    """Label each row with the episode its target window opens onto.

    Spiking intervals are read off the anchors themselves -- every interval in
    the evaluation period appears once as an anchor -- so this needs no table
    the feature mart does not already carry.
    """
    spiking = frame.loc[frame["rrp_now"] > threshold, "anchor_settlement_end"]
    episodes = find_episodes(spiking)
    frame = frame.copy()
    frame["episode_id"] = assign_episodes(frame["anchor_settlement_end"], spiking, episodes)
    return frame, len(episodes)
