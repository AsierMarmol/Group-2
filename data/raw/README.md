# Original market data

Expected input: 54 Binance Spot monthly one-minute CSVs for BTCUSDT, ETHUSDT and SOLUSDT,
April 2025 through September 2026. Keep one subfolder per symbol. Preserve originals unchanged.

main.py resolves SETTINGS and passes the raw-data path and schema explicitly to the loader. ML_RAW_DIR can select an existing raw-data root
containing the three symbol subfolders; do not duplicate large datasets.
Historical timestamps are microseconds; REST timestamps are milliseconds. Normalize both to UTC candle opening times.
Validate filenames, dates, OHLCV, integer trade counts and candle duration. Missing minutes remain gaps;
reject affected input/target windows, and reject conflicting duplicates.

Data files are ignored by Git; this README may be tracked. The local integrated run verified all 54 source files; publisher authenticity is not established by local hashes.
