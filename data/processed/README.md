# Processed data

The integrated pipeline writes clean.csv here after candle cleaning and validation,
using paths from main.SETTINGS. Rows identify an asset and UTC candle opening time.
These are generated outputs: never edit them manually or commit them. Recreate them from the original data.
The local run on 10 November 2026 wrote 2,367,360 clean candles. These are not feature rows or labelled splits.

clean.csv contains cleaned source candles; labels, feature eligibility and split membership are derived later.
Use the metrics report for eligible split counts rather than treating every clean candle as a training sample.
