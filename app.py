"""
app.py — CV Assurance Platform · Streamlit Dashboard

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config
from assurance.dataset_scanner import scan_dataset
from assurance.model_fingerprint import fingerprint_model
from assurance.backdoor_scanner import scan_backdoor, demo_trigger_effect, apply_trigger
from assurance.provenance import (
    create_inference_record,
    verify_entire_chain,
    simulate_tampering,
    get_latest_record,
    clear_chain,
    _load_chain,
)
from assurance.gradcam import explain_image, TARGET_LAYERS, DEFAULT_LAYER
from assurance.report import build_report, LIMITATIONS, SUPPORTED_ATTACKS, NOT_GUARANTEED

# ────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CV Assurance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark cybersecurity theme CSS
st.markdown("""
<style>
/* Global background */
.stApp { background-color: #0d1117; color: #c9d1d9; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #161b22; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Cards */
div.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* Status badges */
.badge-pass   { background:#1a7f37; color:#fff; padding:2px 10px; border-radius:12px; font-weight:700; }
.badge-warn   { background:#9a6700; color:#fff; padding:2px 10px; border-radius:12px; font-weight:700; }
.badge-high   { background:#b62324; color:#fff; padding:2px 10px; border-radius:12px; font-weight:700; }
.badge-info   { background:#1f6feb; color:#fff; padding:2px 10px; border-radius:12px; font-weight:700; }
.badge-na     { background:#30363d; color:#8b949e; padding:2px 10px; border-radius:12px; font-weight:700; }

/* Section headers */
h2.section-title { color:#58a6ff; font-family:monospace; border-bottom:1px solid #21262d; padding-bottom:4px; }

/* Metrics */
div[data-testid="metric-container"] { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:0.5rem; }

/* Finding cards */
.finding-high   { border-left: 4px solid #f85149; padding-left: 10px; margin: 6px 0; }
.finding-medium { border-left: 4px solid #d29922; padding-left: 10px; margin: 6px 0; }
.finding-low    { border-left: 4px solid #1f6feb; padding-left: 10px; margin: 6px 0; }
.finding-pass   { border-left: 4px solid #3fb950; padding-left: 10px; margin: 6px 0; }

/* Hash mono */
.hash-mono { font-family: monospace; font-size: 0.75rem; color: #8b949e; word-break: break-all; }

/* Big status */
.status-verified { font-size:1.8rem; font-weight:900; color:#3fb950; }
.status-tampered { font-size:1.8rem; font-weight:900; color:#f85149; }
.status-warning  { font-size:1.4rem; font-weight:700; color:#d29922; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "selected_dataset": config.DATA_CLEAN_DIR,
        "selected_model":   config.CLEAN_MODEL_PATH,
        "data_scan_result": None,
        "fingerprint_result": None,
        "backdoor_result":  None,
        "chain_result":     None,
        "report":           None,
        "last_inference":   None,
        "inference_info":   None,
        "tampering_done":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ CV Assurance Platform")
    st.markdown("*Evidence-based assurance for computer-vision pipelines*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Dataset Scanner", "🤖 Model Scanner", "🔑 Provenance",
         "🔥 Explainability", "📋 Assurance Report"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("### Dataset")
    dataset_choice = st.selectbox(
        "Active dataset",
        options=[config.DATA_CLEAN_DIR, config.DATA_POISONED_DIR],
        format_func=lambda x: "✅ Clean dataset" if "clean" in x else "⚠️ Poisoned dataset",
    )
    st.session_state["selected_dataset"] = dataset_choice

    st.markdown("### Model")
    model_choice = st.selectbox(
        "Active model",
        options=[config.CLEAN_MODEL_PATH, config.BACKDOORED_MODEL_PATH],
        format_func=lambda x: "✅ Clean model" if "clean" in x else "☠️ Backdoored model",
    )
    st.session_state["selected_model"] = model_choice

    # Check existence
    clean_model_exists      = Path(config.CLEAN_MODEL_PATH).exists()
    backdoored_model_exists = Path(config.BACKDOORED_MODEL_PATH).exists()
    clean_data_exists       = Path(config.DATA_CLEAN_DIR).exists() and any(
        Path(config.DATA_CLEAN_DIR).rglob("*.png"))
    poisoned_data_exists    = Path(config.DATA_POISONED_DIR).exists() and any(
        Path(config.DATA_POISONED_DIR).rglob("*.png"))

    st.markdown("---")
    st.markdown("### Setup Status")
    st.markdown(f"{'✅' if clean_data_exists else '❌'} Clean data")
    st.markdown(f"{'✅' if poisoned_data_exists else '❌'} Poisoned data")
    st.markdown(f"{'✅' if clean_model_exists else '❌'} Clean model")
    st.markdown(f"{'✅' if backdoored_model_exists else '❌'} Backdoored model")

    if not (clean_data_exists and clean_model_exists):
        st.warning("Run setup scripts first — see README.")

    st.markdown("---")
    st.caption("Prototype — offline, SHA-256 only")
    st.caption("Not for production use.")


# ────────────────────────────────────────────────────────────────────────────
# Helper: severity colour
# ────────────────────────────────────────────────────────────────────────────

SEVER_COLOUR = {
    "HIGH":   "#f85149",
    "MEDIUM": "#d29922",
    "LOW":    "#1f6feb",
    "PASS":   "#3fb950",
}

SEVER_EMOJI = {
    "HIGH":   "🔴",
    "MEDIUM": "🟡",
    "LOW":    "🔵",
    "PASS":   "✅",
}


def _score_colour(score: int) -> str:
    if score >= 80: return "#3fb950"
    if score >= 50: return "#d29922"
    return "#f85149"


def _render_finding(f):
    cls_map = {"HIGH": "finding-high", "MEDIUM": "finding-medium",
               "LOW": "finding-low", "PASS": "finding-pass"}
    colour  = SEVER_COLOUR.get(f.severity, "#8b949e")
    css_cls = cls_map.get(f.severity, "finding-low")
    emoji   = SEVER_EMOJI.get(f.severity, "ℹ️")
    with st.container():
        st.markdown(
            f'<div class="{css_cls}">'
            f'<strong style="color:{colour}">{emoji} [{f.severity}] {f.category}</strong><br>'
            f'{f.description}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────────────────────────────────────
# Dashboard page
# ────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🛡️ CV ASSURANCE DASHBOARD</h1>',
                unsafe_allow_html=True)
    st.caption("Evidence-based assurance against supported attack classes · Prototype implementation")

    # Quick overview metrics
    dr = st.session_state["data_scan_result"]
    fr = st.session_state["fingerprint_result"]
    br = st.session_state["backdoor_result"]
    cr = st.session_state["chain_result"]
    rp = st.session_state["report"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = dr.integrity_score if dr else "—"
        st.metric("Data Integrity", f"{score}/100" if dr else "Not scanned")
    with col2:
        mscore = fr and (100 - (0 if fr.binary_match else 40) - (0 if fr.behavioral_match_pct >= 90 else 20))
        st.metric("Model Integrity", f"{rp.model_integrity_score}/100" if rp else "Not scanned")
    with col3:
        pstatus = rp.provenance_status if rp else (
            "VERIFIED" if (cr and cr.is_intact) else ("FAILED" if cr else "N/A"))
        colour = "#3fb950" if pstatus == "VERIFIED" else ("#f85149" if pstatus == "FAILED" else "#8b949e")
        st.markdown(f"**Provenance**<br><span style='color:{colour};font-weight:700;font-size:1.2rem'>{pstatus}</span>",
                    unsafe_allow_html=True)
    with col4:
        asr = br.most_suspicious_asr if br else None
        if asr is not None:
            level = "HIGH" if asr > 0.5 else ("MED" if asr > 0.25 else "LOW")
            colour2 = SEVER_COLOUR.get("HIGH" if level == "HIGH" else "MEDIUM" if level == "MED" else "LOW", "#8b949e")
            st.markdown(f"**Backdoor Risk**<br><span style='color:{colour2};font-weight:700;font-size:1.2rem'>{level} ({asr*100:.0f}% ASR)</span>",
                        unsafe_allow_html=True)
        else:
            st.metric("Backdoor Risk", "Not scanned")

    st.markdown("---")

    # Run all scans button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        run_all = st.button("▶ Run All Scans", type="primary", use_container_width=True)

    if run_all:
        dataset_path  = st.session_state["selected_dataset"]
        model_path    = st.session_state["selected_model"]

        with st.spinner("Running dataset scan …"):
            ref_model = config.CLEAN_MODEL_PATH if Path(config.CLEAN_MODEL_PATH).exists() else None
            dr = scan_dataset(dataset_path, reference_model_path=ref_model)
            st.session_state["data_scan_result"] = dr

        with st.spinner("Running model fingerprint …"):
            if Path(model_path).exists():
                fr = fingerprint_model(model_path)
                st.session_state["fingerprint_result"] = fr
            else:
                fr = None

        with st.spinner("Running backdoor scan …"):
            if Path(model_path).exists():
                br = scan_backdoor(model_path)
                st.session_state["backdoor_result"] = br
            else:
                br = None

        cr = verify_entire_chain()
        st.session_state["chain_result"] = cr

        rp = build_report(
            data_result        = dr,
            fingerprint_result = fr,
            backdoor_result    = br,
            chain_result       = cr,
            dataset_path       = dataset_path,
            model_path         = model_path,
        )
        st.session_state["report"] = rp
        st.success("All scans complete — see sidebar pages for details.")
        st.rerun()

    # Active findings
    if rp:
        st.markdown('<h2 class="section-title">ACTIVE FINDINGS</h2>', unsafe_allow_html=True)
        for f in rp.findings:
            _render_finding(f)

        st.markdown("---")
        status_colour = "#f85149" if "HIGH" in rp.overall_status or "REVIEW" in rp.overall_status else \
                        "#d29922" if "CAUTION" in rp.overall_status else "#3fb950"
        st.markdown(
            f'<div class="card"><strong>Overall Assessment:</strong><br>'
            f'<span style="color:{status_colour};font-size:1.2rem;font-weight:700">{rp.overall_status}</span><br><br>'
            f'<strong>Recommended Disposition:</strong> {rp.recommended_disposition}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Click **▶ Run All Scans** to perform an integrated assessment.")
        st.markdown("""
        ### Demo Flow

        1. Select **Clean dataset** + **Clean model** → run scans → see ✅ all PASS
        2. Switch to **Poisoned dataset** → re-scan → see ⚠️ anomalies
        3. Switch to **Backdoored model** → re-scan → see 🔴 HIGH backdoor risk
        4. Go to **Provenance** page → generate inference → verify → tamper → verify again
        5. Go to **Assurance Report** for the full summary
        """)


# ────────────────────────────────────────────────────────────────────────────
# Dataset Scanner page
# ────────────────────────────────────────────────────────────────────────────

def page_dataset():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">📊 DATASET SCANNER</h1>',
                unsafe_allow_html=True)
    st.caption("Prototype: perceptual-hash duplicate detection + label-consistency anomaly scan")

    dataset_path = st.session_state["selected_dataset"]
    st.info(f"Active dataset: `{dataset_path}`")

    run_btn = st.button("▶ Scan Dataset", type="primary")
    if run_btn:
        ref_model = config.CLEAN_MODEL_PATH if Path(config.CLEAN_MODEL_PATH).exists() else None
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def prog(name, pct):
            progress_bar.progress(pct)
            status_text.text(f"[{pct}%] {name}")

        result = scan_dataset(dataset_path, reference_model_path=ref_model,
                              progress_callback=prog)
        progress_bar.progress(100)
        status_text.text("Done!")
        st.session_state["data_scan_result"] = result

    result = st.session_state["data_scan_result"]
    if result is None:
        st.info("Click **▶ Scan Dataset** to start.")
        return

    if result.scan_error:
        st.error(f"Scan error: {result.scan_error}")
        return

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples",    result.total_samples)
    col2.metric("Duplicate Samples", result.duplicate_count,
                delta=f"-{result.duplicate_count}" if result.duplicate_count else None,
                delta_color="inverse")
    col3.metric("Label Anomalies",   len(result.label_anomalies),
                delta=f"-{len(result.label_anomalies)}" if result.label_anomalies else None,
                delta_color="inverse")
    score = result.integrity_score
    col4.metric("Integrity Score", f"{score}/100")

    # Integrity score gauge
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.barh(0, 100, color="#21262d", height=0.5)
    ax.barh(0, score, color=_score_colour(score), height=0.5)
    ax.set_xlim(0, 100)
    ax.axis("off")
    ax.set_title(f"Data Integrity: {score}/100", color="#c9d1d9", fontsize=10)
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    # Tabs
    tab_dup, tab_anom, tab_contrib = st.tabs(["Duplicates", "Label Anomalies", "Contributors"])

    with tab_dup:
        st.markdown(f"**{result.duplicate_count} images** in **{len(result.duplicate_groups)} duplicate groups** (pHash Hamming ≤ {config.PHASH_DUP_THRESHOLD})")
        if result.duplicate_groups:
            for i, grp in enumerate(result.duplicate_groups[:5]):
                with st.expander(f"Group {i+1} — {1 + len(grp.duplicates)} images (Hamming ≈ {grp.hamming_distance:.1f})"):
                    cols = st.columns(min(1 + len(grp.duplicates), 4))
                    all_imgs = [grp.representative] + grp.duplicates
                    for c_idx, img_path in enumerate(all_imgs[:4]):
                        if Path(img_path).exists():
                            with cols[c_idx]:
                                st.image(img_path, caption=Path(img_path).name,
                                         use_container_width=True)
                    st.markdown(
                        f'<div class="finding-medium">'
                        f'<strong>Finding:</strong> Near-duplicate images<br>'
                        f'<strong>Severity:</strong> MEDIUM<br>'
                        f'<strong>Evidence:</strong> pHash Hamming distance ≈ {grp.hamming_distance:.1f}<br>'
                        f'<strong>Recommendation:</strong> Deduplicate training set; investigate data pipeline for resampling.'
                        f'</div>', unsafe_allow_html=True)
        else:
            st.success("No near-duplicates detected.")

    with tab_anom:
        anomalies = result.label_anomalies
        st.markdown(f"**{len(anomalies)}** label-consistency anomalies found")
        st.caption("⚠️ Disagreement is NOT proof of poisoning — it flags statistical anomalies for review.")
        if anomalies:
            rows = [{
                "File":        Path(a.filename).name,
                "Dataset Label":    a.dataset_label,
                "Model Prediction": a.predicted_label,
                "Confidence":  f"{a.confidence*100:.1f}%",
                "Contributor": a.contributor_id,
            } for a in anomalies[:50]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Show a few examples
            st.markdown("**Example findings:**")
            for a in anomalies[:3]:
                if Path(a.filename).exists():
                    col_img, col_info = st.columns([1, 3])
                    with col_img:
                        st.image(a.filename, use_container_width=True)
                    with col_info:
                        st.markdown(
                            f'<div class="finding-medium">'
                            f'<strong>Finding:</strong> Label-consistency anomaly<br>'
                            f'<strong>Evidence:</strong><br>'
                            f'&nbsp;&nbsp;Dataset label = <code>{a.dataset_label}</code><br>'
                            f'&nbsp;&nbsp;Reference prediction = <code>{a.predicted_label}</code><br>'
                            f'&nbsp;&nbsp;Confidence = {a.confidence:.2f}<br>'
                            f'<strong>Severity:</strong> MEDIUM<br>'
                            f'<strong>Recommendation:</strong> REVIEW SAMPLE — not confirmed poisoning.'
                            f'</div>', unsafe_allow_html=True)
        else:
            st.success("No significant label-consistency anomalies detected.")

    with tab_contrib:
        st.markdown("**Contributor-level risk aggregation**")
        stats = result.contributor_stats
        if stats:
            rows = [{
                "Contributor":  c.contributor_id,
                "Samples":      c.total_samples,
                "Duplicates":   c.duplicate_count,
                "Anomalies":    c.anomaly_count,
                "Risk Score":   f"{c.risk_score*100:.0f}%",
                "Risk Level":   c.risk_level,
            } for c in stats]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Visual bar chart
            fig2, ax2 = plt.subplots(figsize=(6, 2))
            fig2.patch.set_facecolor("#161b22")
            ax2.set_facecolor("#161b22")
            bar_colors = [SEVER_COLOUR.get(c.risk_level, "#8b949e") for c in stats]
            ax2.barh([c.contributor_id for c in stats],
                     [c.risk_score * 100 for c in stats],
                     color=bar_colors)
            ax2.axvline(30, color="#d29922", linestyle="--", linewidth=1, label="MEDIUM threshold")
            ax2.set_xlabel("Risk Score (%)", color="#8b949e")
            ax2.set_title("Contributor Risk Scores", color="#c9d1d9")
            ax2.tick_params(colors="#8b949e")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#30363d")
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("No contributor metadata available.")


# ────────────────────────────────────────────────────────────────────────────
# Model Scanner page
# ────────────────────────────────────────────────────────────────────────────

def page_model():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🤖 MODEL SCANNER</h1>',
                unsafe_allow_html=True)

    model_path = st.session_state["selected_model"]
    st.info(f"Active model: `{model_path}`")

    if not Path(model_path).exists():
        st.error(f"Model file not found: `{model_path}`")
        st.info("Run `python scripts/train_clean_model.py` (and/or train_backdoored_model.py) first.")
        return

    col_fp, col_bd = st.columns([1, 1])

    with col_fp:
        if st.button("▶ Fingerprint Model", type="primary", use_container_width=True):
            with st.spinner("Computing fingerprint …"):
                fr = fingerprint_model(model_path)
                st.session_state["fingerprint_result"] = fr

    with col_bd:
        if st.button("▶ Behavioral Trigger Scan", type="primary", use_container_width=True):
            with st.spinner("Running trigger scan (may take ~30 s) …"):
                br = scan_backdoor(model_path)
                st.session_state["backdoor_result"] = br

    st.markdown("---")

    # Fingerprint results
    fr = st.session_state["fingerprint_result"]
    if fr:
        st.markdown("### 🔐 Model Fingerprint")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model file:** `{fr.model_name}`")
            st.markdown(f"**SHA-256:**")
            st.code(fr.sha256, language=None)
            st.markdown(f"**Reference SHA-256:**")
            st.code(fr.reference_sha256 if fr.reference_sha256 else "(not registered)", language=None)

            binary_icon = "✅ MATCH" if fr.binary_match else "❌ MISMATCH"
            binary_col  = "#3fb950" if fr.binary_match else "#f85149"
            st.markdown(f'<span style="color:{binary_col};font-weight:700">Binary fingerprint: {binary_icon}</span>',
                        unsafe_allow_html=True)

        with col2:
            beh_col = "#3fb950" if fr.behavioral_match_pct >= 90 else "#f85149"
            st.markdown(f'<div class="card">'
                        f'<strong>Behavioral Fingerprint</strong><br>'
                        f'<span style="color:{beh_col};font-size:1.5rem;font-weight:700">{fr.behavioral_match_pct:.1f}%</span> agreement<br>'
                        f'Canary mismatches: <strong>{fr.canary_mismatches}/{fr.canary_total}</strong><br><br>'
                        f'Assessment: <strong>{fr.assessment}</strong>'
                        f'</div>', unsafe_allow_html=True)

        if not fr.binary_match:
            st.error("⚠️ MODEL SUBSTITUTION / MODIFICATION DETECTED — SHA-256 does not match reference.")

        if fr.scan_error:
            st.warning(f"Scan warning: {fr.scan_error}")

    # Backdoor results
    br = st.session_state["backdoor_result"]
    if br:
        st.markdown("---")
        st.markdown("### ☠️ Behavioral Trigger Scan")
        st.caption("Prototype: tests white-square patch at 5 candidate positions. Not a universal backdoor detector.")

        if br.scan_error:
            st.error(f"Scan error: {br.scan_error}")
        else:
            # Results table
            rows = []
            for tr in br.trigger_results:
                flag = " ← SUSPICIOUS" if tr.is_suspicious else ""
                rows.append({
                    "Position":     tr.position,
                    "ASR (%)":      f"{tr.attack_success_rate*100:.0f}%",
                    "Target Class": tr.target_class,
                    "Suspicion":    tr.suspicion_level + flag,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(7, 3))
            fig.patch.set_facecolor("#161b22")
            ax.set_facecolor("#161b22")
            positions = [tr.position for tr in br.trigger_results]
            asrs      = [tr.attack_success_rate * 100 for tr in br.trigger_results]
            colors    = [SEVER_COLOUR["HIGH"] if tr.is_suspicious else "#1f6feb"
                         for tr in br.trigger_results]
            bars = ax.bar(positions, asrs, color=colors)
            ax.axhline(config.SUSPICION_THRESHOLD * 100, color="#d29922",
                       linestyle="--", linewidth=1.5, label=f"Suspicion threshold ({config.SUSPICION_THRESHOLD*100:.0f}%)")
            ax.set_ylabel("Attack Success Rate (%)", color="#8b949e")
            ax.set_title("Trigger Position vs Attack Success Rate", color="#c9d1d9")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")
            for bar, asr in zip(bars, asrs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{asr:.0f}%", ha="center", va="bottom",
                        color="#c9d1d9", fontsize=9)
            st.pyplot(fig, use_container_width=True)

            # Summary
            overall_col = "#f85149" if "HIGH" in br.overall_assessment else \
                          "#d29922" if "MEDIUM" in br.overall_assessment else "#3fb950"
            st.markdown(
                f'<div class="card">'
                f'<strong>Most suspicious position:</strong> {br.most_suspicious_position}<br>'
                f'<strong>Target class:</strong> {br.target_class}<br>'
                f'<strong>Attack success rate:</strong> {br.most_suspicious_asr*100:.0f}%<br>'
                f'<strong>Assessment:</strong> <span style="color:{overall_col};font-weight:700">{br.overall_assessment}</span>'
                f'</div>', unsafe_allow_html=True)

            # Visual demo: normal vs triggered
            st.markdown("#### 🖼️ Trigger Effect Demo")
            # Find a test image
            test_images = list(Path(config.DATA_CLEAN_DIR).rglob("*.png"))
            if not test_images:
                test_images = list(Path(config.CANARY_DIR).rglob("*.png"))

            if test_images:
                demo_img_path = str(test_images[5 % len(test_images)])
                try:
                    demo_results = demo_trigger_effect(
                        model_path, demo_img_path,
                        trigger_position=br.most_suspicious_position or "bottom-left"
                    )
                    col_orig, col_trig = st.columns(2)
                    with col_orig:
                        orig = demo_results["original"]
                        st.image(orig["image_array"], caption=f"Normal: {orig['predicted_label']} ({orig['confidence']*100:.0f}%)",
                                 use_container_width=True)
                        st.markdown(f'<span style="color:#3fb950;font-weight:700">→ {orig["predicted_label"].upper()} ({orig["confidence"]*100:.0f}%)</span>',
                                    unsafe_allow_html=True)
                    with col_trig:
                        trig = demo_results["triggered"]
                        st.image(trig["image_array"], caption=f"Triggered: {trig['predicted_label']} ({trig['confidence']*100:.0f}%)",
                                 use_container_width=True)
                        col = "#f85149" if trig["predicted_label"] == br.target_class else "#d29922"
                        st.markdown(f'<span style="color:{col};font-weight:700">→ {trig["predicted_label"].upper()} ({trig["confidence"]*100:.0f}%)</span>',
                                    unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not run trigger demo: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Provenance page
# ────────────────────────────────────────────────────────────────────────────

def page_provenance():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🔑 INFERENCE PROVENANCE</h1>',
                unsafe_allow_html=True)
    st.caption("Cryptographic SHA-256 hash chain · Fully local · Detects tampering")

    model_path = st.session_state["selected_model"]

    if not Path(model_path).exists():
        st.error(f"No model found at `{model_path}`")
        return

    # Choose test image
    test_images = (
        list(Path(config.DATA_CLEAN_DIR).rglob("*.png")) +
        list(Path(config.CANARY_DIR).rglob("*.png"))
    )

    if not test_images:
        st.error("No test images found. Run `python scripts/generate_demo_data.py` first.")
        return

    demo_img_path = str(test_images[0])

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gen_btn = st.button("⚡ Generate Inference", type="primary", use_container_width=True)
    with col2:
        verify_btn = st.button("🔍 Verify Latest Record", use_container_width=True)
    with col3:
        chain_btn = st.button("🔗 Verify Entire Chain", use_container_width=True)
    with col4:
        tamper_btn = st.button("💀 Simulate Tampering", use_container_width=True)

    st.markdown("---")

    if gen_btn:
        with st.spinner("Running inference and creating provenance record …"):
            try:
                record, info = create_inference_record(demo_img_path, model_path)
                st.session_state["last_inference"] = record
                st.session_state["inference_info"] = info
                st.session_state["tampering_done"] = False
                st.success("Inference record created and appended to chain.")
            except Exception as e:
                st.error(f"Error: {e}")

    if tamper_btn:
        records = _load_chain()
        if not records:
            st.warning("No records in chain. Generate an inference first.")
        else:
            ok = simulate_tampering(record_idx=-1, field="output")
            if ok:
                st.session_state["tampering_done"] = True
                st.error("💀 Record tampered! The output field was modified without updating the hash.")
            else:
                st.warning("Could not tamper — no records found.")

    # Display latest record
    rec = get_latest_record()
    if rec:
        info = st.session_state.get("inference_info")
        tampered = st.session_state.get("tampering_done", False)

        col_img, col_details = st.columns([1, 2])
        with col_img:
            if info:
                st.image(info["image_array"], caption="Input image", use_container_width=True)
                pred = info["predicted_label"]
                conf = info["confidence"]
                st.markdown(f'<div style="text-align:center;font-size:1.3rem;font-weight:700;color:#58a6ff">'
                            f'{pred.upper()}<br><span style="font-size:0.9rem;color:#8b949e">{conf*100:.0f}% confidence</span>'
                            f'</div>', unsafe_allow_html=True)
            else:
                st.caption("(Re-generate inference to see image)")

        with col_details:
            st.markdown("**Latest Inference Record**")
            st.markdown(f"**ID:** `{rec.inference_id}`")
            st.markdown(f"**Output:** `{rec.output}` (confidence: {rec.confidence*100:.1f}%)")
            st.markdown(f"**Timestamp:** `{rec.timestamp}`")
            st.markdown(f"**Input hash:** `{rec.input_hash[:24]}…`")
            st.markdown(f"**Model hash:** `{rec.model_hash[:24]}…`")
            st.markdown(f"**Prev hash:** `{rec.previous_record_hash[:24]}…`")
            st.markdown(f"**Record hash:** `{rec.record_hash[:24]}…`")

    # Verify buttons
    if verify_btn or chain_btn:
        if verify_btn and rec:
            from assurance.provenance import verify_inference_record
            vr = verify_inference_record(rec)
            if vr.is_valid:
                st.markdown('<p class="status-verified">✓ RECORD VERIFIED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-tampered">✗ INFERENCE TAMPERED</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="finding-high">'
                    f'<strong>Expected hash:</strong><br>'
                    f'<span class="hash-mono">{vr.expected_hash}</span><br><br>'
                    f'<strong>Observed hash:</strong><br>'
                    f'<span class="hash-mono">{vr.observed_hash}</span><br><br>'
                    f'<strong>Broken field:</strong> {vr.broken_field}'
                    f'</div>', unsafe_allow_html=True)

        if chain_btn:
            cr = verify_entire_chain()
            st.session_state["chain_result"] = cr
            if cr.is_intact:
                st.markdown('<p class="status-verified">✓ CHAIN INTACT</p>', unsafe_allow_html=True)
                st.markdown(f"All **{cr.total_records}** records verified ✅")
            else:
                st.markdown('<p class="status-tampered">✗ CHAIN INTEGRITY VIOLATED</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="finding-high">'
                    f'<strong>Broken at record:</strong> {cr.first_broken}<br>'
                    f'<strong>Record ID:</strong> {cr.broken_record_id}<br>'
                    f'{cr.total_records - cr.valid_records} record(s) invalid out of {cr.total_records}'
                    f'</div>', unsafe_allow_html=True)

                for vr in cr.verification_details:
                    if not vr.is_valid:
                        st.markdown(
                            f'<div class="finding-high">'
                            f'<strong>Record {vr.record_id[:8]}… FAILED</strong><br>'
                            f'Expected: <span class="hash-mono">{vr.expected_hash}</span><br>'
                            f'Observed: <span class="hash-mono">{vr.observed_hash}</span><br>'
                            f'{vr.error_msg}'
                            f'</div>', unsafe_allow_html=True)

    # Chain history table
    st.markdown("---")
    st.markdown("**Chain History**")
    records = _load_chain()
    if records:
        rows = [{
            "ID":        r.inference_id[:8] + "…",
            "Output":    r.output,
            "Conf":      f"{r.confidence*100:.0f}%",
            "Timestamp": r.timestamp[:10],
            "Hash":      r.record_hash[:12] + "…",
        } for r in records[-10:]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No inference records yet. Click **⚡ Generate Inference**.")

    col_clr, _ = st.columns([1, 5])
    with col_clr:
        if st.button("🗑️ Clear Chain", use_container_width=True):
            clear_chain()
            st.session_state["last_inference"] = None
            st.session_state["inference_info"] = None
            st.session_state["tampering_done"] = False
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Assurance Report page
# ────────────────────────────────────────────────────────────────────────────

def page_report():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">📋 ASSURANCE REPORT</h1>',
                unsafe_allow_html=True)

    rp = st.session_state["report"]

    if not rp:
        st.info("No report generated yet. Go to **Dashboard** and click **▶ Run All Scans**.")
        # Auto-generate from existing state
        cr = verify_entire_chain()
        st.session_state["chain_result"] = cr
        return

    # Header
    status_colour = ("#f85149" if "HIGH" in rp.overall_status or "REVIEW" in rp.overall_status
                     else "#d29922" if "CAUTION" in rp.overall_status else "#3fb950")
    st.markdown(
        f'<div class="card">'
        f'<h2 style="color:{status_colour};font-family:monospace">CV ASSURANCE REPORT</h2>'
        f'<strong>Overall Status:</strong> <span style="color:{status_colour};font-weight:700">{rp.overall_status}</span><br>'
        f'<strong>Dataset:</strong> {rp.dataset_path}<br>'
        f'<strong>Model:</strong> {rp.model_path}'
        f'</div>', unsafe_allow_html=True)

    # Scores
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Integrity",  f"{rp.data_integrity_score}/100")
    col2.metric("Model Integrity", f"{rp.model_integrity_score}/100")
    prov_col = "#3fb950" if rp.provenance_status == "VERIFIED" else "#f85149"
    col3.markdown(f"**Provenance**<br><span style='color:{prov_col};font-weight:700;font-size:1.3rem'>{rp.provenance_status}</span>",
                  unsafe_allow_html=True)

    # Score bars
    fig, axes = plt.subplots(1, 2, figsize=(8, 1))
    fig.patch.set_facecolor("#0d1117")
    for ax, (label, score) in zip(axes, [
        ("Data Integrity", rp.data_integrity_score),
        ("Model Integrity", rp.model_integrity_score),
    ]):
        ax.set_facecolor("#0d1117")
        ax.barh(0, 100, color="#21262d", height=0.5)
        ax.barh(0, score, color=_score_colour(score), height=0.5)
        ax.set_xlim(0, 100)
        ax.axis("off")
        ax.set_title(f"{label}: {score}/100", color="#c9d1d9", fontsize=9)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Findings")
    if rp.findings:
        for f in rp.findings:
            _render_finding(f)
            with st.expander(f"Evidence & Recommendation — {f.category}"):
                st.markdown(f"**Evidence:** {f.evidence}")
                st.markdown(f"**Recommendation:** {f.recommendation}")
    else:
        st.success("No significant findings.")

    st.markdown("---")
    st.markdown(
        f'<div class="card">'
        f'<strong>Recommended Disposition:</strong><br>'
        f'<span style="font-size:1.1rem;color:#58a6ff">{rp.recommended_disposition}</span>'
        f'</div>', unsafe_allow_html=True)

    # Limitations (very important)
    st.markdown("---")
    st.markdown("### ⚠️ Scope & Limitations")
    st.caption("This section is important for technical credibility.")

    col_supp, col_ng = st.columns(2)
    with col_supp:
        st.markdown("**Supported attack classes:**")
        for a in SUPPORTED_ATTACKS:
            st.markdown(f"✅ {a}")
    with col_ng:
        st.markdown("**NOT guaranteed to detect:**")
        for a in NOT_GUARANTEED:
            st.markdown(f"❌ {a}")

    st.markdown("**Limitations:**")
    for lim in LIMITATIONS:
        st.markdown(f"- {lim}")

    # Export JSON
    st.markdown("---")
    if st.button("💾 Export Report JSON"):
        report_data = {
            "overall_status": rp.overall_status,
            "data_integrity_score": rp.data_integrity_score,
            "model_integrity_score": rp.model_integrity_score,
            "provenance_status": rp.provenance_status,
            "recommended_disposition": rp.recommended_disposition,
            "findings": [{"severity": f.severity, "category": f.category,
                          "description": f.description, "recommendation": f.recommendation}
                         for f in rp.findings],
            "limitations": LIMITATIONS,
            "supported_attacks": SUPPORTED_ATTACKS,
        }
        st.download_button(
            "📥 Download report.json",
            data=json.dumps(report_data, indent=2),
            file_name="assurance_report.json",
            mime="application/json",
        )



# ────────────────────────────────────────────────────────────────────────────
# Page: Explainability (Grad-CAM)
# ────────────────────────────────────────────────────────────────────────────

def _upscale(arr: np.ndarray, size: int = 256) -> np.ndarray:
    """Nearest-neighbour upscale so 32×32 images stay crisp on screen."""
    return np.array(Image.fromarray(arr).resize((size, size), Image.NEAREST))


def _heatmap_rgb(heatmap: np.ndarray, size: int = 256) -> np.ndarray:
    hm = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR))
    return (plt.get_cmap("jet")(hm / 255.0)[..., :3] * 255).astype(np.uint8)


def page_explainability():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🔥 GRAD-CAM EXPLAINABILITY</h1>',
                unsafe_allow_html=True)
    st.caption("Shows which image regions drove the active model's prediction")

    model_path = st.session_state["selected_model"]
    model_name = "Clean model" if model_path == config.CLEAN_MODEL_PATH else "Backdoored model"
    if not Path(model_path).exists():
        st.error(f"No model found at `{model_path}`")
        return
    st.info(f"Active model: **{model_name}** (change it in the sidebar)")

    # ── Input image ──────────────────────────────────────────────────────────
    col_src, col_opts = st.columns([3, 2])
    with col_src:
        source = st.radio("Image source", ["Upload an image", "Pick a sample"], horizontal=True)
        image = None
        if source == "Upload an image":
            up = st.file_uploader("Image (resized to 32×32)", type=["png", "jpg", "jpeg", "bmp", "webp"])
            if up is not None:
                image = Image.open(up).convert("RGB")
        else:
            samples = sorted(Path(config.DATA_CLEAN_DIR).rglob("*.png")) + sorted(Path(config.CANARY_DIR).glob("*.png"))
            if not samples:
                st.error("No sample images found. Run `python scripts/generate_demo_data.py` first.")
                return
            choice = st.selectbox("Sample", samples,
                                  format_func=lambda p: f"{p.parent.name}/{p.name}")
            image = Image.open(choice).convert("RGB")

    with col_opts:
        stamp = st.checkbox("Stamp backdoor trigger patch",
                            help="Adds the 5×5 white patch used by the demo backdoor attack.")
        position = st.selectbox("Trigger position", config.TRIGGER_CANDIDATES,
                                index=config.TRIGGER_CANDIDATES.index(config.TRIGGER_POSITION),
                                disabled=not stamp)
        explain_opts = ["Predicted class"] + config.CLASS_NAMES
        explain = st.selectbox("Class to explain", explain_opts)
        layer = st.selectbox("Target layer", list(TARGET_LAYERS),
                             index=list(TARGET_LAYERS).index(DEFAULT_LAYER))
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)

    if image is None:
        st.info("Upload an image or pick a sample to see the explanation.")
        return

    if stamp:
        arr = np.array(image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
        image = Image.fromarray(apply_trigger(arr, position))

    target = None if explain == "Predicted class" else config.CLASS_NAMES.index(explain)
    try:
        res = explain_image(model_path, image, target_class=target, layer_idx=TARGET_LAYERS[layer])
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
        return

    st.markdown("---")

    # ── Prediction + maps ────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:1.3rem;font-weight:700;color:#58a6ff">'
                f'Prediction: {res.predicted_label.upper()} '
                f'<span style="font-size:0.95rem;color:#8b949e">({res.confidence*100:.1f}% confidence)</span>'
                f'</div>', unsafe_allow_html=True)
    if res.explained_idx != res.predicted_idx:
        st.caption(f"Heatmap explains **{res.explained_label}** "
                   f"({res.probs[res.explained_idx]*100:.1f}% probability), not the predicted class.")

    input_big = _upscale(res.image)
    heat_big  = _heatmap_rgb(res.heatmap)
    overlay   = ((1 - alpha) * input_big + alpha * heat_big).astype(np.uint8)

    c1, c2, c3 = st.columns(3)
    c1.image(input_big, caption="Model input (32×32)", use_container_width=True)
    c2.image(heat_big,  caption=f"Grad-CAM for '{res.explained_label}'", use_container_width=True)
    c3.image(overlay,   caption="Overlay", use_container_width=True)
    st.caption("Red = regions that most increased the score for the explained class; blue = little influence.")

    # ── Class probabilities ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 2.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    colours = ["#58a6ff" if i == res.predicted_idx else "#30363d" for i in range(len(config.CLASS_NAMES))]
    ax.bar(config.CLASS_NAMES, res.probs * 100, color=colours)
    ax.set_ylabel("Probability (%)", color="#c9d1d9")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Backdoor hint ────────────────────────────────────────────────────────
    if stamp:
        ts, h = config.TRIGGER_SIZE, config.IMAGE_SIZE
        rows = {"top-left": (0, 0), "top-right": (0, h - ts), "center": (h//2 - ts//2, h//2 - ts//2),
                "bottom-left": (h - ts, 0), "bottom-right": (h - ts, h - ts)}
        r, c = rows[position]
        patch_focus = float(res.heatmap[r:r + ts, c:c + ts].mean())
        overall     = float(res.heatmap.mean())
        if patch_focus > 2 * overall and res.predicted_idx == config.BACKDOOR_TARGET:
            st.markdown(
                f'<div class="card finding-high"><strong>Finding:</strong> Attention concentrated on trigger patch<br>'
                f'<strong>Evidence:</strong> Mean activation on patch {patch_focus:.2f} vs {overall:.2f} image-wide; '
                f'prediction flipped to <strong>{res.predicted_label}</strong>.<br>'
                f'<strong>Interpretation:</strong> Consistent with a patch-triggered backdoor.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card finding-pass"><strong>Trigger patch does not dominate the prediction.</strong><br>'
                f'Mean activation on patch {patch_focus:.2f} vs {overall:.2f} image-wide.</div>',
                unsafe_allow_html=True)

    st.caption("⚠️ Grad-CAM is a coarse, gradient-based approximation (8×8 or 16×16 grid upsampled). "
               "It shows correlation with the class score, not proof of causation.")

# ────────────────────────────────────────────────────────────────────────────
# Router
# ────────────────────────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    page_dashboard()
elif page == "📊 Dataset Scanner":
    page_dataset()
elif page == "🤖 Model Scanner":
    page_model()
elif page == "🔑 Provenance":
    page_provenance()
elif page == "🔥 Explainability":
    page_explainability()
elif page == "📋 Assurance Report":
    page_report()

