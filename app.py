"""Streamlit front end — Tooth-Level Selective Prediction demo.

Run (from this folder, caches stay on D:):
    run_demo.bat
or:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
sys.path.insert(0, THIS_DIR)

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from inference import (
    ACCEPT,
    DEFER,
    CAVITY,
    BUNDLE,
    ScreeningEngine,
)

SAMPLE_DIR = BUNDLE.samples
FIG_DIR = BUNDLE.figures

_PORTABLE_SAMPLES = [
    ("Sample A — CariXray test", "sample_A.jpg"),
    ("Sample B — CariXray test", "sample_B.jpg"),
    ("Sample C — CariXray test", "sample_C.jpg"),
    ("Sample D — CariXray test", "sample_D.jpg"),
]
_REPO_SAMPLES = [
    ("Sample A — CariXray test", "test_crawling_104_jpg.rf.27c9d15caf42c9dc786200b01354bebe.jpg"),
    ("Sample B — CariXray test", "test_crawling_128_jpg.rf.6f8e4a5957c8ae6a79325fb1859c7d50.jpg"),
    ("Sample C — CariXray test", "test_crawling_160_jpg.rf.1708dbb29b9bf596895413f8e5e4d5ee.jpg"),
    ("Sample D — CariXray test", "test_crawling_167_jpg.rf.8476d496bdf31002ecb318d858e41a59.jpg"),
]
SAMPLES = _PORTABLE_SAMPLES if BUNDLE.portable else _REPO_SAMPLES

st.set_page_config(
    page_title="TRIDENT — Tooth-Level Selective Screening",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: "Source Sans 3", sans-serif; }
.stApp { background: #0f1419; color: #e8eef4; }

#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(105deg, #0b3a5b 0%, #123047 55%, #1a1f26 100%);
    border: 1px solid #1e4a6a;
    border-radius: 14px;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    margin-bottom: 1.1rem;
}
.hero h1 {
    font-size: 1.55rem; font-weight: 700; margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em; color: #f4f8fb;
}
.hero p { margin: 0; color: #b7c7d6; font-size: 0.95rem; }
.badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    background: #d55e00; color: white; padding: 0.18rem 0.55rem;
    border-radius: 999px; margin-bottom: 0.55rem;
}

.metric-row { display: flex; gap: 0.7rem; margin: 0.4rem 0 1rem 0; }
.mcard {
    flex: 1; background: #182029; border: 1px solid #2a3846;
    border-radius: 10px; padding: 0.7rem 0.85rem;
}
.mcard .k { font-size: 0.72rem; color: #8aa0b3; text-transform: uppercase; letter-spacing: 0.04em; }
.mcard .v { font-size: 1.45rem; font-weight: 700; color: #f4f8fb; font-family: "IBM Plex Mono", monospace; }
.mcard.accept { border-color: #0072b2; }
.mcard.defer  { border-color: #d55e00; }
.mcard.cav    { border-color: #c44; }

.legend span { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
               font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; }
.lg-clear  { background: #0072b2; color: white; }
.lg-cav    { background: #c0392b; color: white; }
.lg-review { background: #d55e00; color: white; }

.note {
    background: #1a2430; border-left: 3px solid #d55e00;
    padding: 0.65rem 0.85rem; border-radius: 0 8px 8px 0;
    color: #c5d2de; font-size: 0.88rem; margin: 0.6rem 0 1rem 0;
}
.oknote {
    background: #132433; border-left: 3px solid #0072b2;
    padding: 0.65rem 0.85rem; border-radius: 0 8px 8px 0;
    color: #c5d2de; font-size: 0.88rem; margin: 0.6rem 0 1rem 0;
}
div[data-testid="stMetric"] { background: #182029; border: 1px solid #2a3846;
    border-radius: 10px; padding: 0.6rem 0.8rem; }
div[data-testid="stMetric"] label { color: #8aa0b3 !important; }
[data-testid="stSidebar"] { background: #12181f; }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    background: #182029; color: #c5d2de; border-radius: 8px 8px 0 0;
    padding: 0.45rem 0.9rem;
}
.stTabs [aria-selected="true"] { background: #0b3a5b; color: white; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLO localizer + 5-seed Swin-T ensemble…")
def get_engine() -> ScreeningEngine:
    return ScreeningEngine()


def load_bgr_from_upload(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img


def load_bgr_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def metric_row(res):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Teeth detected", res.n_teeth)
    c2.metric("Accepted", res.n_accept)
    c3.metric("Deferred (review)", res.n_defer)
    c4.metric("Accepted · cavity", res.n_cavity_accepted)
    c5.metric("Accepted · clear", res.n_clear_accepted)


def render_tooth_cards(res):
    if not res.teeth:
        return
    st.markdown("##### Per-tooth crops")
    cols = st.columns(min(6, len(res.teeth)))
    for i, t in enumerate(res.teeth):
        with cols[i % len(cols)]:
            rgb = cv2.cvtColor(t.crop_bgr, cv2.COLOR_BGR2RGB)
            cap = f"#{t.index}  r={t.r:.2f}  {t.decision}"
            st.image(rgb, caption=cap, use_container_width=True)
            if t.decision == DEFER:
                st.caption(f":orange[REVIEW] · P={t.p_cal:.2f}")
            elif t.label == CAVITY:
                st.caption(f":red[CAVITY] · P={t.p_cal:.2f}")
            else:
                st.caption(f":blue[CLEAR] · P={t.p_cal:.2f}")


# ── header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="badge">Research demo · not a medical device</div>
  <h1>Tooth-Level Selective Prediction for Risk-Aware Dental Caries Screening</h1>
  <p>Radiograph → YOLOv8n tooth localisation → 5-seed Swin-T → temperature scaling (T = 1.768)
     → validation-fitted risk score → <b>accept</b> or <b>defer</b>. Locked 2026 protocol.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_live, tab_results, tab_method = st.tabs(
    ["Live screening", "Locked paper results", "How it works"]
)

# ── sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Decision rule")
    st.caption(
        "Population τ is taken from the frozen **internal validation** "
        "risk-coverage curve (the paper’s operating-point definition). "
        "It is not fitted on the image you upload."
    )
    mode = st.radio(
        "Operating point",
        ["≤5% accepted-case risk (paper)", "≤10% accepted-case risk", "Manual τ"],
        index=0,
    )
    manual_tau = None
    risk_target = 0.05
    if mode.startswith("≤5%"):
        risk_target = 0.05
    elif mode.startswith("≤10%"):
        risk_target = 0.10
    else:
        risk_target = None
        manual_tau = st.slider("τ  (accept if r ≥ τ)", 0.05, 0.95, 0.50, 0.01)

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown(
        '<div class="legend">'
        '<span class="lg-clear">ACCEPT · clear</span>'
        '<span class="lg-cav">ACCEPT · cavity</span>'
        '<span class="lg-review">DEFER · review</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(
        "Selector features (locked): confidence, entropy H, "
        "epistemic u, localisation q. Coefficients from "
        "`calibration/learned_selector.json`."
    )


# ── LIVE ────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown(
        '<div class="note"><b>Disclaimer.</b> Demonstration of a research pipeline. '
        "Tooth localisation was trained on classical-CV pseudo-labels, not expert tooth boxes. "
        "Do not use for diagnosis or treatment decisions.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])
    with left:
        uploaded = st.file_uploader(
            "Upload a periapical radiograph (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
    with right:
        sample_labels = [s[0] for s in SAMPLES]
        sample_choice = st.selectbox(
            "Or run a CariXray test-split sample",
            ["—"] + sample_labels,
        )

    image_bgr = None
    source_name = None
    if uploaded is not None:
        try:
            image_bgr = load_bgr_from_upload(uploaded)
            source_name = uploaded.name
        except Exception as exc:
            st.error(str(exc))
    elif sample_choice != "—":
        fname = dict(SAMPLES)[sample_choice]
        path = os.path.join(SAMPLE_DIR, fname)
        if os.path.exists(path):
            image_bgr = load_bgr_from_path(path)
            source_name = sample_choice
        else:
            st.warning(f"Sample not found at {path}")

    run = st.button("Run screening", type="primary", disabled=image_bgr is None)

    if image_bgr is not None and not run and "last_result" not in st.session_state:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Input · {source_name} · {image_bgr.shape[1]}×{image_bgr.shape[0]}",
            use_container_width=True,
        )

    if run and image_bgr is not None:
        try:
            engine = get_engine()
            with st.spinner("Localising teeth and running the 5-seed ensemble…"):
                res = engine.screen(
                    image_bgr,
                    tau=manual_tau,
                    risk_target=risk_target,
                )
            st.session_state["last_result"] = res
            st.session_state["last_source"] = source_name
        except Exception as exc:
            st.exception(exc)
            res = None
    else:
        res = st.session_state.get("last_result")

    if res is not None:
        st.markdown(
            f'<div class="oknote">τ = <b>{res.tau:.3f}</b>'
            + (
                f" &nbsp;·&nbsp; risk target α = <b>{100 * res.risk_target:.0f}%</b> (from internal val)"
                if res.risk_target is not None and manual_tau is None
                else " &nbsp;·&nbsp; manual threshold"
            )
            + f" &nbsp;·&nbsp; source: {st.session_state.get('last_source', '')}</div>",
            unsafe_allow_html=True,
        )
        metric_row(res)
        c_img, c_tab = st.columns([1.35, 1])
        with c_img:
            st.image(res.overlay_rgb, caption="Accept / defer overlay", use_container_width=True)
        with c_tab:
            if res.teeth:
                df = pd.DataFrame([t.as_row() for t in res.teeth])
                st.dataframe(df, use_container_width=True, hide_index=True)
                n = max(res.n_teeth, 1)
                st.caption(
                    f"Referral rate on this radiograph: "
                    f"{100 * res.n_defer / n:.0f}%  ({res.n_defer}/{res.n_teeth} teeth). "
                    "This is a per-image rate, not the paper’s population coverage."
                )
            else:
                st.info("No teeth detected above the localiser threshold.")
        render_tooth_cards(res)


# ── RESULTS ─────────────────────────────────────────────────────────────
with tab_results:
    st.markdown("##### Numbers locked in `FINAL_RESULTS_INDEX.md` — not recomputed in this demo.")
    a, b, c, d = st.columns(4)
    a.metric("Internal AURC (confidence → proposed)", "0.0436 → 0.0389")
    b.metric("Coverage at ≤5% risk", "63.7% → 71.1%")
    c.metric("Swin-T 5-seed AUC", "0.949 ± 0.002")
    d.metric("External coverage at ≤5% risk", "≤ 0.25%")
    st.caption(
        "Coverage 63.7% is raw confidence; 71.1% is 5-seed disagreement; "
        "proposed selector reaches 67.9% at the same 5% risk target. "
        "n = 1812 matched internal teeth / 2028 external."
    )

    f1 = os.path.join(FIG_DIR, "Fig1_TRIDENT_framework_600dpi.png")
    f2 = os.path.join(FIG_DIR, "Fig2_internal_selective_prediction_600dpi.png")
    f3 = os.path.join(FIG_DIR, "Fig3_selective_risk_transportability_600dpi.png")
    if os.path.exists(f1):
        st.markdown("**Fig. 1 — Pipeline**")
        st.image(f1, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(f2):
            st.markdown("**Fig. 2 — In-distribution selective prediction**")
            st.image(f2, use_container_width=True)
    with c2:
        if os.path.exists(f3):
            st.markdown("**Fig. 3 — Transportability (does not generalise)**")
            st.image(f3, use_container_width=True)

    st.markdown(
        '<div class="note">The in-distribution coverage gain <b>does not transport</b> to '
        "Caries-ETSKL. That is a result of the paper, not a bug in this demo.</div>",
        unsafe_allow_html=True,
    )


# ── METHOD ──────────────────────────────────────────────────────────────
with tab_method:
    st.markdown(
        """
**What this page runs**

1. **Localise** each tooth with the frozen YOLOv8n detector (`imgsz = 320`).
2. **Crop** with 8% padding.
3. **Classify** with a 5-seed Swin-T ensemble (seeds 42–46). Seed 42 is the
   original capstone checkpoint; 43–46 were trained for this study.
4. **Calibrate** the ensemble-mean P(cavity) with temperature **T = 1.768**,
   fit on internal validation only.
5. **Score** each tooth
   \(r_i = \sigma(w^\\top [c_i, H_i, u_i, q_i^{\\mathrm{norm}}] + b)\)
   using the stored logistic weights (P(correct)).
6. **Accept** if \(r_i \ge \\tau\), otherwise **defer** to review.
   Default \(\\tau\) is the validation operating point at 5% accepted-case risk.

**What is not claimed as novel:** YOLOv8, Swin-T, temperature scaling, ensembles.

**What is demonstrated:** tooth-instance accept/defer with a validation-fitted
multi-signal score, evaluated in the paper with risk-coverage / AURC.

**Do not mix protocols.** The older capstone app (`Capastone Project/src/app.py`)
uses fixed 0.80 / 0.50 dual thresholds. This demo does not.
        """
    )
