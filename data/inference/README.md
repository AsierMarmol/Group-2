# Inference inputs

The local batch pipeline produces historical diagnostic predictions. Raw-candle replay is also available
through `src.infer.score_candles`; no API or polling service is implemented in this deliverable.

## Public raw-candle contract

Supply a pandas DataFrame with exactly 60 completed consecutive one-minute candles for one asset:

| Column | Required value |
| --- | --- |
| symbol | The same nonmissing BTCUSDT, ETHUSDT or SOLUSDT on every row |
| open_time | Candle opening time: integer microseconds/milliseconds or timezone-aware datetimes; UTC minute boundaries |
| open, high, low, close | Finite positive prices with low/high consistent with open and close |
| volume | Finite nonnegative base volume |
| trades | Finite nonnegative integer trade count |

The historical loader calls the trade-count field `trade_count`; rename it to `trades` for this raw contract.
Order rows by opening time, with no duplicate or missing minutes. Missing symbols, incomplete/invalid windows,
naive timestamps and unsupported assets raise an error instead of silently producing a score.
`as_of` must explicitly declare a timezone and be at or after the final input candle closes.
Choose `unit="us"` for historical integers or `unit="ms"` for REST integers; datetimes must include a timezone.

```python
from pathlib import Path
from src import infer

# window is the caller-supplied DataFrame satisfying the contract above.
score = infer.score_candles(
    window,
    Path("models/candidates/model.joblib"),
    as_of="2026-09-30T23:45:00+00:00",
    unit="ms",
)
```

The result contains symbol, risk_score, decision_time, model_version, horizon_minutes, mode, status,
calibration_status and approval_status. risk_score is uncalibrated; replay does not approve an artifact for live use.
The lower-level live-policy checks require an explicitly approved artifact and freshness limit; they do not
implement a live service or define the team's still-pending alert/freshness policy.

Demo rows may overlap training and the final period has prior exposure: neither is independent confirmation.
Input datasets stay outside Git. Final automated checks cover the public contract and its negative cases.
