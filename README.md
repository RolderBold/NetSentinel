# NetSentinel

NetSentinel is a small Streamlit dashboard for the UNSW-NB15 autoencoder project. It takes a CSV of network-flow feature rows, calculates an autoencoder reconstruction error for each row, and flags scores above the saved threshold.

This dashboard analyzes uploaded CSV files. It does not capture packets or monitor a live network. A flag means a flow looks unusual compared with the benign examples used for training; it is a review signal, not proof of an attack.

## Project contents

```text
NetSentinel/
├── app.py
├── requirements.txt
├── models/
│   ├── netsentinel_autoencoder.keras
│   └── netsentinel_preprocessing.joblib
└── notebooks/
    └── NetSentinel_Anomaly_Detection.ipynb
```

## Run on Windows

Install Python 3.11, open PowerShell in this folder, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens the dashboard in your browser. The app uses TensorFlow on CPU; no GPU is needed. The first dependency installation is several hundred MB because of TensorFlow. The UNSW-NB15 dataset is not included in this project folder and does not need to be stored locally in full.

TensorFlow's Windows installation is CPU-only for current releases. The model was trained with TensorFlow 2.21.0, so the requirement file uses that version for compatibility.

## Prepare a small demo CSV in Colab

The dashboard expects the same 42 feature columns used in the notebook. Extra columns such as `id`, `label`, or `attack_cat` are accepted. The labels are excluded from model inputs; if `label` is present, it is used only to show evaluation metrics.

In the already-run Colab notebook, run this cell to download a small labeled sample for a demo:

```python
demo_csv = test_df.sample(n=500, random_state=42)
demo_csv.to_csv("/content/netsentinel_demo_flows.csv", index=False)

from google.colab import files
files.download("/content/netsentinel_demo_flows.csv")
```

Upload `netsentinel_demo_flows.csv` in the dashboard. The dashboard will show the flagged rows, anomaly scores, a downloadable scored CSV, and test metrics because this demo sample includes its known labels.

## Input and interpretation

- Required inputs: all feature columns fitted by the notebook's saved preprocessor.
- Optional reference columns: `id`, binary `label` (0 = normal, 1 = attack), and `attack_cat`.
- Blank required feature values are rejected so the app does not silently score incomplete records.
- The anomaly score is mean squared reconstruction error. It is not a probability or confidence score.
- The app keeps the uploaded CSV in memory for the current dashboard session and does not save it to this project folder.

## Dataset citation

The notebook and results use UNSW-NB15. See the [official UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for dataset details and academic citations.
