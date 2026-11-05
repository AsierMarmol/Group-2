# Notebook roles

The notebook owner's later PR will provide:
- crypto_volatility_ml.ipynb: the frozen standalone ML reference.
- volatility_model.ipynb: modular execution, saved-model reload and parity checks.

Keep the original reference unchanged. New experiments belong in a separate notebook;
the production pipeline ports final choices and does not tune on the final evaluation period.
Use the team Python 3.12 kernel with its working directory at the repository root; do not modify sys.path.
ML_RAW_DIR may select an existing data root. ML_PROJECT_DIR may select the standalone notebook's project root.
Record actual execution results when notebooks are integrated. Do not commit datasets or generated model files.
