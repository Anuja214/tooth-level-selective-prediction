# TRIDENT laptop demo

Tooth-level selective caries screening (2026 protocol).

**Not a medical device.** Research demonstration only.

## What is in this zip

- Streamlit app (`app.py`)
- YOLOv8n tooth localiser + 5-seed Swin-T checkpoints
- Locked calibration (T = 1.768) and selector weights
- 4 CariXray test-split sample radiographs
- Paper figures 1–3

**Not included** (too large / not needed to run the demo):
full CariXray / ETSKL datasets, training logs, archive YOLO dumps, venv.

## Setup (Windows)

1. Unzip anywhere (example: `C:\Users\you\TRIDENT_laptop_demo` or a D: folder).
2. Install **Python 3.10 or 3.11** from python.org and tick “Add Python to PATH”.
3. Open a terminal in this folder:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

4. Install PyTorch (pick one):

CPU laptop:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU laptop:

```bat
pip install torch torchvision
```

5. Then:

```bat
pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `run_demo.bat` after the venv is activated.

Open **http://localhost:8501**

First run downloads Ultralytics extras if needed and loads five Swin-T models (slow). Later runs are faster.

## Tabs

- **Live screening** — upload a JPG/PNG or pick a sample
- **Locked paper results** — in-distribution vs external numbers
- **How it works** — protocol

Default decision: accept if risk score r ≥ τ, with τ from internal validation at ≤5% accepted-case risk.

## Hardware

Works on CPU. A GPU is optional. 8 GB RAM is enough; 16 GB is more comfortable.
