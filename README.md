# Medical AI Prediction Web - 15 Feature Version

This folder contains the Streamlit deployment package for the 15-feature model.

## Files

- `web.py` - Streamlit app entry point.
- `requirements.txt` - Python dependencies pinned for the saved model.
- `runtime.txt` and `.python-version` - Python 3.12 runtime settings.
- `data.xlsx` - Data file used to infer input ranges and categorical options.
- `src/ml_project/` - Project source code imported by the app.
- `outputs_20260524_003517/models/best_model.joblib` - 15-feature model loaded by the app.
- `outputs_20260524_003517/models/best_model_metadata.json` - 15-feature list, threshold, and model metadata.

## Run

```bash
pip install -r requirements.txt
streamlit run web.py
```

