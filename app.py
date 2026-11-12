"""
app.py — CV Assurance Platform · Streamlit Dashboard

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config
from assurance.datasets import (
    BATCH_COLUMNS, CONTRIBUTOR_COLUMNS, LABEL_COLUMNS, PATH_COLUMNS, TORCHVISION_DATASETS,
    guess_column, load_csv, load_folder, load_torchvision, read_table, table_columns,
)
from assurance.dataset_scanner import ScanParams, scan_dataset
from assurance.features import EXTRACTORS
from assurance.model_loader import (
    BUILTIN_ARCHS, TORCHVISION_ARCHS, ModelSpec, load_model, read_sidecar, sidecar_path, write_sidecar,
)
from assurance.model_fingerprint import (
    build_canary_set, fingerprint_model, list_canary_manifests, list_references,
    load_canary_set, register_reference,
)
from assurance.backdoor_scanner import (
    POSITIONS, apply_trigger, default_trigger_size, demo_trigger_effect, scan_backdoor, trigger_origin,
)
from assurance.provenance import (
    _load_chain, clear_chain, create_inference_record, get_latest_record,
    simulate_tampering, verify_entire_chain, verify_inference_record,
)
from assurance.gradcam import explain_image, list_target_layers
from assurance.report import LIMITATIONS, NOT_GUARANTEED, SUPPORTED_ATTACKS, build_report

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
.stApp { background-color: #0d1117; color: #c9d1d9; }
section[data-testid="stSidebar"] { background-color: #161b22; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
div.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
h2.section-title { color:#58a6ff; font-family:monospace; border-bottom:1px solid #21262d; padding-bottom:4px; }
div[data-testid="metric-container"] { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:0.5rem; }
.finding-high   { border-left: 4px solid #f85149; padding-left: 10px; margin: 6px 0; }
.finding-medium { border-left: 4px solid #d29922; padding-left: 10px; margin: 6px 0; }
.finding-low    { border-left: 4px solid #1f6feb; padding-left: 10px; margin: 6px 0; }
.finding-pass   { border-left: 4px solid #3fb950; padding-left: 10px; margin: 6px 0; }
.hash-mono { font-family: monospace; font-size: 0.75rem; color: #8b949e; word-break: break-all; }
.status-verified { font-size:1.8rem; font-weight:900; color:#3fb950; }
.status-tampered { font-size:1.8rem; font-weight:900; color:#f85149; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────

RESULT_KEYS = ("data_scan_result", "fingerprint_result", "backdoor_result", "chain_result", "report")


def _init_state():
    defaults = {
        "dataset": None,             # assurance.datasets.Dataset
        "model": None,               # assurance.model_loader.LoadedModel
        "scan_params": ScanParams(),
        "reference_name": None,
        "last_inference": None,
        "inference_info": None,
        "tampering_done": False,
        "_bootstrapped": False,
        **{k: None for k in RESULT_KEYS},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # First run: pre-load the demo assets if present, so the demo works out of the box
    if not st.session_state["_bootstrapped"]:
        st.session_state["_bootstrapped"] = True
        demo_ds = next(iter(config.DEMO_DATASETS.values()))
        demo_m = next(iter(config.DEMO_MODELS.values()))
        try:
            if Path(demo_ds).is_dir():
                st.session_state["dataset"] = load_folder(demo_ds, name="Demo · clean")
            if Path(demo_m).exists() and sidecar_path(demo_m).exists():
                st.session_state["model"] = load_model(ModelSpec(path=demo_m, name="Demo · clean model"))
        except Exception:
            pass
        refs = list_references()
        if refs:
            st.session_state["reference_name"] = next(iter(refs))


def _set_dataset(ds):
    st.session_state["dataset"] = ds
    for k in ("data_scan_result", "backdoor_result", "report"):
        st.session_state[k] = None


def _set_model(m):
    st.session_state["model"] = m
    for k in ("fingerprint_result", "backdoor_result", "report", "inference_info"):
        st.session_state[k] = None


_init_state()


def _save_upload(upload) -> str:
    """Persist a Streamlit upload under outputs/uploads/ and return its path."""
    data = upload.getvalue()
    d = Path(config.UPLOADS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{hashlib.sha256(data).hexdigest()[:12]}_{Path(upload.name).name}"
    if not p.exists():
        p.write_bytes(data)
    return str(p)


def _need_dataset():
    ds = st.session_state["dataset"]
    if ds is None:
        st.warning("No dataset loaded — go to **⚙️ Setup** to ingest one.")
    return ds


def _need_model():
    m = st.session_state["model"]
    if m is None:
        st.warning("No model loaded — go to **⚙️ Setup** to load one.")
    return m


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ CV Assurance Platform")
    st.markdown("*Evidence-based assurance for computer-vision pipelines*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "⚙️ Setup", "📊 Dataset Scanner", "🤖 Model Scanner", "🔑 Provenance",
         "🔥 Explainability", "📋 Assurance Report"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    ds = st.session_state["dataset"]
    m = st.session_state["model"]
    st.markdown("### Active dataset")
    st.markdown(f"✅ **{ds.name}**  \n{len(ds)} images · {len(ds.class_names)} classes" if ds else "❌ none")
    st.markdown("### Active model")
    st.markdown(f"✅ **{m.name}**  \n{m.arch} · {m.num_classes} classes · {m.input_size[0]}×{m.input_size[1]}"
                if m else "❌ none")
    if ds and m and set(ds.class_names) != set(m.class_names):
        st.warning("Dataset and model class names differ.")

    demo_ds = {k: v for k, v in config.DEMO_DATASETS.items() if Path(v).is_dir()}
    demo_m = {k: v for k, v in config.DEMO_MODELS.items() if Path(v).exists()}
    if demo_ds or demo_m:
        with st.expander("Demo shortcuts"):
            for label, path in demo_ds.items():
                if st.button(f"📁 {label}", use_container_width=True, key=f"demo_{label}"):
                    _set_dataset(load_folder(path, name=label))
                    st.rerun()
            for label, path in demo_m.items():
                if st.button(f"🤖 {label}", use_container_width=True, key=f"demo_{label}"):
                    try:
                        _set_model(load_model(ModelSpec(path=path, name=label)))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    st.markdown("---")
    st.caption("Prototype — offline, SHA-256 only")
    st.caption("Not for production use.")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

SEVER_COLOUR = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#1f6feb", "PASS": "#3fb950"}
SEVER_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "PASS": "✅"}


def _score_colour(score: int) -> str:
    if score >= 80: return "#3fb950"
    if score >= 50: return "#d29922"
    return "#f85149"


def _render_finding(f):
    cls_map = {"HIGH": "finding-high", "MEDIUM": "finding-medium", "LOW": "finding-low", "PASS": "finding-pass"}
    colour = SEVER_COLOUR.get(f.severity, "#8b949e")
    st.markdown(
        f'<div class="{cls_map.get(f.severity, "finding-low")}">'
        f'<strong style="color:{colour}">{SEVER_EMOJI.get(f.severity, "ℹ️")} [{f.severity}] {f.category}</strong><br>'
        f'{f.description}</div>', unsafe_allow_html=True)


def _rel(path: str, root: str) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return Path(path).name


def _dark_axes(ax, fig):
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def _sample_paths(ds, limit=500):
    return [r.filename for r in ds.records[:limit]]


# ────────────────────────────────────────────────────────────────────────────
# Setup page: dataset ingestion + model loading
# ────────────────────────────────────────────────────────────────────────────

def _column_picker(rows, prefix, need_label=False):
    """Selectboxes to map table columns → path / label / contributor / batch."""
    cols = table_columns(rows)
    none = "(none)"

    def pick(label, candidates, required):
        guess = guess_column(cols, candidates)
        opts = cols if required else [none] + cols
        idx = opts.index(guess) if guess in opts else 0
        v = st.selectbox(label, opts, index=idx, key=f"{prefix}_{label}")
        return None if v == none else v

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        path_col = pick("File/path column", PATH_COLUMNS, True)
    with c2:
        label_col = pick("Label column", LABEL_COLUMNS, True) if need_label else None
    with c3:
        contrib_col = pick("Contributor column", CONTRIBUTOR_COLUMNS, False)
    with c4:
        batch_col = pick("Batch column", BATCH_COLUMNS, False)
    st.caption(f"{len(rows)} rows · columns: {', '.join(cols)}")
    return path_col, label_col, contrib_col, batch_col


def _setup_dataset():
    st.markdown('<h2 class="section-title">1 · DATASET</h2>', unsafe_allow_html=True)
    source = st.radio("Source", ["Folder of class sub-folders", "CSV + image files", "torchvision dataset"],
                      horizontal=True)

    try:
        if source == "Folder of class sub-folders":
            st.caption("Layout: `root/<class_name>/<image>` — class names are the folder names. "
                       "Contributor / batch info comes from an optional metadata CSV/JSONL with one row per image.")
            root = st.text_input("Dataset root folder", placeholder="/path/to/dataset")
            meta_mode = st.radio("Metadata", ["Auto-detect (metadata.csv / metadata.jsonl in root)",
                                              "Metadata file path", "Upload metadata", "None"], horizontal=True)
            meta, cols = None, (None, None, None, None)
            if meta_mode == "Metadata file path":
                mp = st.text_input("Metadata path (.csv / .jsonl)")
                if mp and Path(mp).is_file():
                    meta = mp
            elif meta_mode == "Upload metadata":
                up = st.file_uploader("Metadata (.csv / .jsonl)", type=["csv", "jsonl", "ndjson"])
                if up is not None:
                    meta = up
            if meta is not None:
                cols = _column_picker(read_table(meta), "fmeta")
            if st.button("📥 Load folder dataset", type="primary", disabled=not root):
                with st.spinner("Indexing images …"):
                    ds = load_folder(root, metadata=meta, metadata_key_col=cols[0],
                                     contributor_col=cols[2], batch_col=cols[3],
                                     auto_metadata=meta_mode.startswith("Auto"))
                _set_dataset(ds)
                st.success(f"Loaded {ds.summary()}")

        elif source == "CSV + image files":
            st.caption("A table with one row per image: a path column (absolute, or relative to the image root) "
                       "and a label column. Optional contributor / batch columns.")
            mode = st.radio("Label file", ["Path", "Upload"], horizontal=True)
            table = None
            if mode == "Path":
                csv_path = st.text_input("Label file path (.csv / .jsonl)")
                if csv_path and Path(csv_path).is_file():
                    table = csv_path
            else:
                up = st.file_uploader("Label file (.csv / .jsonl)", type=["csv", "jsonl", "ndjson"])
                if up is not None:
                    table = up
            image_root = st.text_input("Image root folder (relative paths resolve against this; "
                                       "defaults to the label file's folder)")
            if table is not None:
                path_col, label_col, contrib_col, batch_col = _column_picker(read_table(table), "csv", need_label=True)
                if st.button("📥 Load CSV dataset", type="primary"):
                    with st.spinner("Indexing images …"):
                        ds = load_csv(table, image_root=image_root or None, path_col=path_col,
                                      label_col=label_col, contributor_col=contrib_col, batch_col=batch_col)
                    _set_dataset(ds)
                    st.success(f"Loaded {ds.summary()}")

        else:
            st.caption("Downloads once into `outputs/torchvision_raw/`, then writes a stratified subset as PNGs "
                       "to `outputs/datasets/` so every scanner can use it. No contributor metadata.")
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.selectbox("Dataset", list(TORCHVISION_DATASETS))
            with c2:
                splits = {"SVHN": ["train", "test", "extra"]}.get(name, ["train", "test"])
                split = st.selectbox("Split", splits)
            with c3:
                limit = st.number_input("Max images (0 = all)", 0, 200_000, 2000, step=500)
            download = st.checkbox("Download if not present", value=True)
            if st.button("📥 Load torchvision dataset", type="primary"):
                bar = st.progress(0.0)
                ds = load_torchvision(name, split, limit=int(limit) or None, download=download,
                                      progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
                bar.progress(1.0, text="Done")
                _set_dataset(ds)
                st.success(f"Loaded {ds.summary()}")
    except Exception as e:
        st.error(f"Could not load dataset: {type(e).__name__}: {e}")

    ds = st.session_state["dataset"]
    if ds is None:
        return
    st.markdown(f"**Active:** `{ds.name}` · source `{ds.source}` · root `{ds.root}`"
                + (f" · metadata `{ds.metadata_path}`" if ds.metadata_path else ""))
    for n in ds.notes:
        st.warning(n)
    c1, c2 = st.columns([3, 2])
    with c1:
        counts = ds.class_counts()
        st.dataframe(pd.DataFrame({"class": list(counts), "images": list(counts.values())}),
                     use_container_width=True, hide_index=True, height=min(35 * len(counts) + 38, 320))
    with c2:
        if ds.has_contributors:
            cc = ds.contributor_counts()
            st.dataframe(pd.DataFrame({"contributor": list(cc), "images": list(cc.values())}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No contributor metadata — contributor risk analysis will be skipped.")
    samples = ds.records[:: max(1, len(ds) // 8)][:8]
    cols = st.columns(len(samples))
    for c, r in zip(cols, samples):
        c.image(r.filename, caption=r.label, use_container_width=True)


def _parse_floats(text: str):
    text = text.strip()
    return [float(x) for x in text.replace(",", " ").split()] if text else None


def _setup_model():
    st.markdown('<h2 class="section-title">2 · MODEL</h2>', unsafe_allow_html=True)
    st.caption("PyTorch `state_dict` + architecture, TorchScript, or a fully pickled `nn.Module`. "
               "Settings are pre-filled from a sidecar JSON next to the file (`model.pt` → `model.json`) if present.")
    src = st.radio("Model file", ["Path", "Upload"], horizontal=True)
    path = None
    if src == "Path":
        p = st.text_input("Model path (.pt / .pth / .ts)")
        path = p if p and Path(p).is_file() else None
        if p and not path:
            st.error("File not found.")
    else:
        up = st.file_uploader("Model file", type=["pt", "pth", "ts", "bin", "ckpt"])
        if up is not None:
            path = _save_upload(up)
    side = read_sidecar(path) if path else {}
    if side:
        st.success(f"Sidecar found: `{sidecar_path(path).name}` → {', '.join(side)}")

    c1, c2 = st.columns(2)
    with c1:
        fmt = st.selectbox("Format", ["auto", "state_dict", "torchscript", "pickled"])
        arch_opts = (["(from sidecar)"] if side.get("arch") else []) + BUILTIN_ARCHS + TORCHVISION_ARCHS + ["custom"]
        arch = st.selectbox("Architecture (state_dict only)", arch_opts,
                            help="TorchScript and pickled models carry their own architecture.")
        custom_py = custom_class = None
        custom_kwargs = {}
        if arch == "custom":
            custom_py = st.text_input("Python file defining the model class")
            custom_class = st.text_input("Class name")
            kw = st.text_input("Constructor kwargs (JSON)", "{}")
            try:
                custom_kwargs = json.loads(kw or "{}")
            except json.JSONDecodeError:
                st.error("kwargs must be valid JSON")
        allow_pickle = st.checkbox("Trust this file (allow pickle)",
                                   help="Needed for torch.save(model) files. Unpickling can execute arbitrary "
                                        "code — only enable for files you trust.")
    with c2:
        ds = st.session_state["dataset"]
        cn_opts = (["From sidecar"] if side.get("class_names") else []) + \
                  (["Same as active dataset"] if ds else []) + ["Enter manually", "Generic (class_0 …)"]
        cn_mode = st.selectbox("Class names (in output-index order)", cn_opts)
        class_names = None
        if cn_mode == "Same as active dataset":
            class_names = list(ds.class_names)
            st.caption(f"{len(class_names)} classes, sorted: {', '.join(class_names[:8])}"
                       f"{' …' if len(class_names) > 8 else ''}")
        elif cn_mode == "Enter manually":
            txt = st.text_area("One per line or comma-separated")
            class_names = [c.strip() for c in txt.replace(",", "\n").splitlines() if c.strip()] or None
        with st.expander("Preprocessing (blank = sidecar / architecture default)"):
            h = st.number_input("Input height", 0, 4096, 0)
            w = st.number_input("Input width", 0, 4096, 0)
            mean = _parse_floats(st.text_input("Normalisation mean (e.g. 0.485 0.456 0.406)"))
            std = _parse_floats(st.text_input("Normalisation std (e.g. 0.229 0.224 0.225)"))

    if st.button("🤖 Load model", type="primary", disabled=path is None):
        spec = ModelSpec(
            path=path, format=fmt, arch=None if arch == "(from sidecar)" else arch,
            custom_py=custom_py, custom_class=custom_class, custom_kwargs=custom_kwargs,
            class_names=None if cn_mode == "From sidecar" else class_names,
            input_size=(int(h), int(w)) if h and w else None, mean=mean, std=std,
            allow_pickle=allow_pickle, name=Path(path).name,
        )
        try:
            with st.spinner("Loading and dry-running model …"):
                _set_model(load_model(spec))
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Could not load model: {type(e).__name__}: {e}")

    m = st.session_state["model"]
    if m is None:
        return
    st.markdown(
        f'<div class="card"><strong>{m.name}</strong> · format <code>{m.format}</code> · arch <code>{m.arch}</code><br>'
        f'{m.num_classes} classes · input {m.input_size[0]}×{m.input_size[1]} · mean {list(m.mean)} · std {list(m.std)}<br>'
        f'<span class="hash-mono">SHA-256 {m.sha256}</span></div>', unsafe_allow_html=True)
    for n in m.notes:
        st.info(n)
    ds = st.session_state["dataset"]
    if ds and set(ds.class_names) != set(m.class_names):
        st.warning("Model class names don't match the active dataset's — the backdoor scan will fall back to "
                   "clean predictions as reference labels.")
    if m.format == "state_dict" and st.button("💾 Save these settings as sidecar JSON"):
        arch_name = m.arch if not m.arch.startswith("custom:") else "custom"
        p = write_sidecar(m.path, arch=arch_name, class_names=m.class_names,
                          input_size=list(m.input_size), mean=list(m.mean), std=list(m.std))
        st.success(f"Wrote {p}")


def page_setup():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">⚙️ DATA & MODEL SETUP</h1>',
                unsafe_allow_html=True)
    st.caption("Everything the scanners know about classes, contributors and preprocessing comes from here.")
    _setup_dataset()
    st.markdown("---")
    _setup_model()


# ────────────────────────────────────────────────────────────────────────────
# Dashboard page
# ────────────────────────────────────────────────────────────────────────────

def _run_all():
    ds, m = st.session_state["dataset"], st.session_state["model"]
    dr = fr = br = None
    if ds is not None:
        with st.spinner("Running dataset scan …"):
            dr = scan_dataset(ds, st.session_state["scan_params"])
            st.session_state["data_scan_result"] = dr
    if m is not None:
        with st.spinner("Running model fingerprint …"):
            fr = fingerprint_model(m, st.session_state["reference_name"])
            st.session_state["fingerprint_result"] = fr
        with st.spinner("Running backdoor scan …"):
            br = scan_backdoor(m, ds)
            st.session_state["backdoor_result"] = br
    cr = verify_entire_chain()
    st.session_state["chain_result"] = cr
    st.session_state["report"] = build_report(
        data_result=dr, fingerprint_result=fr, backdoor_result=br, chain_result=cr,
        dataset_path=ds.root if ds else "", model_path=m.path if m else "")


def page_dashboard():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🛡️ CV ASSURANCE DASHBOARD</h1>',
                unsafe_allow_html=True)
    st.caption("Evidence-based assurance against supported attack classes · Prototype implementation")

    dr = st.session_state["data_scan_result"]
    br = st.session_state["backdoor_result"]
    cr = st.session_state["chain_result"]
    rp = st.session_state["report"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Integrity", f"{dr.integrity_score}/100" if dr else "Not scanned")
    col2.metric("Model Integrity", f"{rp.model_integrity_score}/100" if rp else "Not scanned")
    with col3:
        pstatus = rp.provenance_status if rp else ("VERIFIED" if (cr and cr.is_intact) else ("FAILED" if cr else "N/A"))
        colour = "#3fb950" if pstatus == "VERIFIED" else ("#f85149" if pstatus == "FAILED" else "#8b949e")
        st.markdown(f"**Provenance**<br><span style='color:{colour};font-weight:700;font-size:1.2rem'>{pstatus}</span>",
                    unsafe_allow_html=True)
    with col4:
        if br and not br.scan_error:
            asr = br.most_suspicious_asr
            level = "HIGH" if asr > br.suspicion_threshold else ("MEDIUM" if asr > 0.25 else "LOW")
            st.markdown(f"**Backdoor Risk**<br><span style='color:{SEVER_COLOUR[level]};font-weight:700;"
                        f"font-size:1.2rem'>{level} ({asr*100:.0f}% ASR)</span>", unsafe_allow_html=True)
        else:
            st.metric("Backdoor Risk", "Not scanned")

    st.markdown("---")
    ds, m = st.session_state["dataset"], st.session_state["model"]
    if ds is None and m is None:
        st.info("Load a dataset and/or model in **⚙️ Setup** first.")
    col_btn1, _ = st.columns([1, 3])
    with col_btn1:
        run_all = st.button("▶ Run All Scans", type="primary", use_container_width=True,
                            disabled=ds is None and m is None)
    if run_all:
        _run_all()
        st.success("All scans complete — see sidebar pages for details.")
        st.rerun()

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
            f'<strong>Recommended Disposition:</strong> {rp.recommended_disposition}</div>',
            unsafe_allow_html=True)
    else:
        st.info("Click **▶ Run All Scans** to perform an integrated assessment.")
        st.markdown("""
        ### Demo Flow
        1. Demo shortcuts (sidebar): **Demo · clean** + **Demo · clean model** → run scans → ✅ all PASS
        2. Switch to **Demo · poisoned** → re-scan → ⚠️ duplicates + label anomalies
        3. Switch to **Demo · backdoored model** → re-scan → 🔴 HIGH backdoor risk, fingerprint mismatch
        4. **Provenance** → generate inference → verify → tamper → verify again
        5. **Assurance Report** for the full summary

        ### Your own data
        **⚙️ Setup** → load a folder / CSV / torchvision dataset and your model → register it as a
        trusted reference on **🤖 Model Scanner** → run scans.
        """)


# ────────────────────────────────────────────────────────────────────────────
# Dataset Scanner page
# ────────────────────────────────────────────────────────────────────────────

def _scan_settings():
    p: ScanParams = st.session_state["scan_params"]
    with st.expander("⚙️ Scan settings"):
        c1, c2, c3 = st.columns(3)
        with c1:
            check = st.checkbox("Check label consistency", value=p.check_labels)
            ext_keys = list(EXTRACTORS)
            extractor = st.selectbox("Feature extractor", ext_keys, index=ext_keys.index(p.extractor),
                                     format_func=EXTRACTORS.get,
                                     help="Pretrained on ImageNet — independent of your data and model.")
        with c2:
            methods = {"knn": "k-nearest-neighbour vote", "linear_probe": "Out-of-fold linear probe"}
            method = st.selectbox("Anomaly method", list(methods), index=list(methods).index(p.anomaly_method),
                                  format_func=methods.get)
            k = st.slider("k neighbours", 3, 50, p.k_neighbors, disabled=method != "knn")
        with c3:
            thr = st.slider("Agreement threshold", 0.3, 1.0, float(p.agreement_threshold), 0.05,
                            help="Flag when at least this share of neighbours (or probe probability) "
                                 "favours another class.")
            dup = st.slider("Duplicate Hamming threshold (of 256 bits)", 0, 64, p.dup_threshold)
        st.session_state["scan_params"] = ScanParams(
            dup_threshold=dup, hash_size=p.hash_size, extractor=extractor, anomaly_method=method,
            k_neighbors=k, agreement_threshold=thr, check_labels=check)


def page_dataset():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">📊 DATASET SCANNER</h1>',
                unsafe_allow_html=True)
    st.caption("Perceptual-hash duplicate detection + label-consistency check in an independent feature space")

    ds = _need_dataset()
    if ds is None:
        return
    st.info(f"Active dataset: **{ds.summary()}**")
    _scan_settings()

    if st.button("▶ Scan Dataset", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def prog(name, pct):
            progress_bar.progress(min(int(pct), 100))
            status_text.text(f"[{int(pct)}%] {name}")

        st.session_state["data_scan_result"] = scan_dataset(ds, st.session_state["scan_params"], prog)
        progress_bar.progress(100)
        status_text.text("Done!")

    result = st.session_state["data_scan_result"]
    if result is None:
        st.info("Click **▶ Scan Dataset** to start.")
        return
    if result.scan_error:
        st.error(f"Scan error: {result.scan_error}")
        return
    for n in result.notes:
        st.warning(n)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", result.total_samples)
    col2.metric("Duplicate Samples", result.duplicate_count)
    col3.metric("Label Anomalies", len(result.label_anomalies))
    col4.metric("Integrity Score", f"{result.integrity_score}/100")

    score = result.integrity_score
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

    tab_dup, tab_anom, tab_contrib = st.tabs(["Duplicates", "Label Anomalies", "Contributors"])

    with tab_dup:
        thr = result.params.dup_threshold if result.params else config.PHASH_DUP_THRESHOLD
        st.markdown(f"**{result.duplicate_count} images** in **{len(result.duplicate_groups)} duplicate groups** "
                    f"(pHash Hamming ≤ {thr})")
        if result.duplicate_groups:
            for i, grp in enumerate(result.duplicate_groups[:10]):
                with st.expander(f"Group {i+1} — {1 + len(grp.duplicates)} images (Hamming ≈ {grp.hamming_distance:.1f})"):
                    all_imgs = [grp.representative] + grp.duplicates
                    cols = st.columns(min(len(all_imgs), 4))
                    for c_idx, img_path in enumerate(all_imgs[:4]):
                        with cols[c_idx]:
                            st.image(img_path, caption=_rel(img_path, result.dataset_path), use_container_width=True)
                    st.markdown(
                        f'<div class="finding-medium"><strong>Finding:</strong> Near-duplicate images<br>'
                        f'<strong>Evidence:</strong> pHash Hamming distance ≈ {grp.hamming_distance:.1f}<br>'
                        f'<strong>Recommendation:</strong> Deduplicate training set; investigate data pipeline for resampling.'
                        f'</div>', unsafe_allow_html=True)
        else:
            st.success("No near-duplicates detected.")

    with tab_anom:
        anomalies = result.label_anomalies
        p = result.params
        st.markdown(f"**{len(anomalies)}** label-consistency anomalies found "
                    f"(extractor: `{result.extractor_used or '—'}`, method: `{p.anomaly_method if p else '—'}`)")
        st.caption("⚠️ Each flagged image looks more like another class than its own label, according to a "
                   "pretrained feature extractor that never saw this dataset. NOT proof of poisoning.")
        if anomalies:
            support = "Neighbour agreement" if p and p.anomaly_method == "knn" else "Probe probability"
            rows = [{
                "File": _rel(a.filename, result.dataset_path),
                "Dataset label": a.dataset_label,
                "Suggested label": a.predicted_label,
                support: f"{a.confidence*100:.0f}%",
                "Support for own label": f"{a.own_label_support*100:.0f}%",
                "Contributor": a.contributor_id,
            } for a in anomalies[:200]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("**Example findings:**")
            for a in anomalies[:3]:
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    st.image(a.filename, use_container_width=True)
                with col_info:
                    st.markdown(
                        f'<div class="finding-medium"><strong>Finding:</strong> Label-consistency anomaly<br>'
                        f'&nbsp;&nbsp;Dataset label = <code>{a.dataset_label}</code><br>'
                        f'&nbsp;&nbsp;Similar images are mostly <code>{a.predicted_label}</code> ({a.confidence:.0%})<br>'
                        f'<strong>Recommendation:</strong> REVIEW SAMPLE — not confirmed poisoning.</div>',
                        unsafe_allow_html=True)
        elif p and not p.check_labels:
            st.info("Label-consistency check disabled in scan settings.")
        else:
            st.success("No significant label-consistency anomalies detected.")

    with tab_contrib:
        st.markdown("**Contributor-level risk aggregation**")
        stats = result.contributor_stats
        if stats:
            st.dataframe(pd.DataFrame([{
                "Contributor": c.contributor_id, "Samples": c.total_samples,
                "Duplicates": c.duplicate_count, "Anomalies": c.anomaly_count,
                "Risk Score": f"{c.risk_score*100:.0f}%", "Risk Level": c.risk_level,
            } for c in stats]), use_container_width=True, hide_index=True)
            fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.35 * len(stats))))
            _dark_axes(ax2, fig2)
            ax2.barh([c.contributor_id for c in stats], [c.risk_score * 100 for c in stats],
                     color=[SEVER_COLOUR.get(c.risk_level, "#8b949e") for c in stats])
            ax2.axvline(30, color="#d29922", linestyle="--", linewidth=1)
            ax2.set_xlabel("Risk Score (%)", color="#8b949e")
            ax2.set_title("Contributor Risk Scores", color="#c9d1d9")
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("No contributor metadata in this dataset. Add a metadata CSV/JSONL with a contributor "
                    "column in **⚙️ Setup** to enable this analysis.")


# ────────────────────────────────────────────────────────────────────────────
# Model Scanner page
# ────────────────────────────────────────────────────────────────────────────

def _reference_panel(m):
    refs = list_references()
    names = list(refs)
    cur = st.session_state["reference_name"]
    if names:
        sel = st.selectbox("Trusted reference", names, index=names.index(cur) if cur in names else 0,
                           format_func=lambda n: f"{n} · {refs[n].get('arch', '?')} · registered {refs[n].get('registered_at', '?')}")
        st.session_state["reference_name"] = sel
    else:
        st.session_state["reference_name"] = None
        st.info("No trusted reference registered yet — register the model you trust below.")

    with st.expander("➕ Register the active model as a trusted reference"):
        st.caption("Stores its SHA-256 and its outputs on a canary set. Later audits replay the same canary "
                   "images; ideally they were held out of training.")
        name = st.text_input("Reference name", Path(m.name).stem)
        ds = st.session_state["dataset"]
        manifests = list_canary_manifests()
        opts = (["Build from active dataset"] if ds else []) + [f"Existing: {Path(p).name}" for p in manifests]
        if not opts:
            st.warning("Load a dataset to build a canary set.")
            return
        choice = st.radio("Canary set", opts)
        per_class = seed = None
        if choice == "Build from active dataset":
            c1, c2 = st.columns(2)
            per_class = c1.number_input("Images per class", 1, 100, config.CANARY_PER_CLASS)
            seed = c2.number_input("Seed", 0, 10**9, config.RANDOM_SEED)
        if st.button("Register reference", disabled=not name):
            try:
                if choice == "Build from active dataset":
                    canary = build_canary_set(ds, per_class=int(per_class), seed=int(seed),
                                              name=f"{name}-canary")
                else:
                    canary = load_canary_set(manifests[opts.index(choice) - (1 if ds else 0)])
                if name in refs:
                    st.warning(f"Overwrote existing reference '{name}'.")
                register_reference(name, m, canary)
                st.session_state["reference_name"] = name
                st.success(f"Registered '{name}' with canary set '{canary.name}' ({len(canary)} images).")
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")


def page_model():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🤖 MODEL SCANNER</h1>', unsafe_allow_html=True)
    m = _need_model()
    if m is None:
        return
    ds = st.session_state["dataset"]
    st.info(f"Active model: **{m.name}** ({m.arch}, {m.num_classes} classes)")

    col_fp, col_bd = st.columns(2)
    with col_fp:
        st.markdown("#### 🔐 Fingerprint")
        _reference_panel(m)
        if st.button("▶ Fingerprint Model", type="primary", use_container_width=True):
            with st.spinner("Computing fingerprint …"):
                st.session_state["fingerprint_result"] = fingerprint_model(m, st.session_state["reference_name"])

    with col_bd:
        st.markdown("#### ☠️ Behavioral trigger scan")
        with st.expander("⚙️ Trigger scan settings"):
            tgt_opts = ["Auto-detect (scan all classes)"] + m.class_names
            tgt = st.selectbox("Target class", tgt_opts)
            ts_default = default_trigger_size(m.input_size)
            ts = st.slider("Trigger patch size (px at model input)", 1, max(4, min(m.input_size) // 3), ts_default)
            n_imgs = st.slider("Test images", 20, 500, 90, 10)
            thr = st.slider("HIGH suspicion threshold (ASR)", 0.1, 0.95, config.SUSPICION_THRESHOLD, 0.05)
        if ds is None:
            st.caption("No dataset loaded — the scan will use random-noise images (weak).")
        if st.button("▶ Behavioral Trigger Scan", type="primary", use_container_width=True):
            with st.spinner("Running trigger scan …"):
                st.session_state["backdoor_result"] = scan_backdoor(
                    m, ds, target_class=None if tgt == tgt_opts[0] else m.class_names.index(tgt),
                    trigger_size=ts, n_test_images=n_imgs, suspicion_threshold=thr)

    st.markdown("---")
    fr = st.session_state["fingerprint_result"]
    if fr:
        st.markdown("### 🔐 Model Fingerprint")
        if fr.scan_error:
            st.error(f"Fingerprint error: {fr.scan_error}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model file:** `{fr.model_name}`")
            st.markdown("**SHA-256:**")
            st.code(fr.sha256, language=None)
            st.markdown(f"**Reference SHA-256** ({fr.reference_name or 'none'}):")
            st.code(fr.reference_sha256 or "(no reference registered)", language=None)
            if fr.has_reference:
                ok = fr.binary_match
                st.markdown(f'<span style="color:{"#3fb950" if ok else "#f85149"};font-weight:700">'
                            f'Binary fingerprint: {"✅ MATCH" if ok else "❌ MISMATCH"}</span>', unsafe_allow_html=True)
        with col2:
            if fr.has_reference and not fr.scan_error:
                beh_col = "#3fb950" if fr.behavioral_match_pct >= 90 else "#f85149"
                st.markdown(
                    f'<div class="card"><strong>Behavioral Fingerprint</strong> · canary set <code>{fr.canary_set}</code><br>'
                    f'<span style="color:{beh_col};font-size:1.5rem;font-weight:700">{fr.behavioral_match_pct:.1f}%</span> top-1 agreement<br>'
                    f'Canary mismatches: <strong>{fr.canary_mismatches}/{fr.canary_total}</strong><br>'
                    f'Mean output shift (total variation): <strong>{fr.mean_prob_shift:.3f}</strong><br>'
                    f'Canary images intact: <strong>{"yes" if fr.canary_intact else "NO"}</strong><br><br>'
                    f'Assessment: <strong>{fr.assessment}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card">Assessment: <strong>{fr.assessment}</strong></div>',
                            unsafe_allow_html=True)
        if fr.has_reference and not fr.binary_match:
            st.error("⚠️ MODEL SUBSTITUTION / MODIFICATION DETECTED — SHA-256 does not match reference.")

    br = st.session_state["backdoor_result"]
    if br:
        st.markdown("---")
        st.markdown("### ☠️ Behavioral Trigger Scan")
        st.caption(f"White {br.trigger_size}×{br.trigger_size} px patch at {len(POSITIONS)} positions on "
                   f"{br.n_test_images} correctly-classified test images. Not a universal backdoor detector.")
        for n in br.notes:
            st.warning(n)
        if br.scan_error:
            st.error(f"Scan error: {br.scan_error}")
            return
        st.dataframe(pd.DataFrame([{
            "Position": tr.position, "ASR (%)": f"{tr.attack_success_rate*100:.0f}%",
            "Most-flipped-to class": tr.target_class,
            "Suspicion": tr.suspicion_level + (" ← SUSPICIOUS" if tr.is_suspicious else ""),
        } for tr in br.trigger_results]), use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        _dark_axes(ax, fig)
        asrs = [tr.attack_success_rate * 100 for tr in br.trigger_results]
        bars = ax.bar([tr.position for tr in br.trigger_results], asrs,
                      color=[SEVER_COLOUR["HIGH"] if tr.is_suspicious else "#1f6feb" for tr in br.trigger_results])
        ax.axhline(br.suspicion_threshold * 100, color="#d29922", linestyle="--", linewidth=1.5,
                   label=f"Suspicion threshold ({br.suspicion_threshold*100:.0f}%)")
        ax.set_ylabel("Attack Success Rate (%)", color="#8b949e")
        ax.set_title("Trigger Position vs Attack Success Rate", color="#c9d1d9")
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")
        for bar, asr in zip(bars, asrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{asr:.0f}%",
                    ha="center", va="bottom", color="#c9d1d9", fontsize=9)
        st.pyplot(fig, use_container_width=True)

        overall_col = "#f85149" if "HIGH" in br.overall_assessment else \
                      "#d29922" if "MEDIUM" in br.overall_assessment else "#3fb950"
        st.markdown(
            f'<div class="card"><strong>Most suspicious position:</strong> {br.most_suspicious_position}<br>'
            f'<strong>Target class:</strong> {br.target_class}'
            f'{"" if br.target_was_given else " (discovered)"}<br>'
            f'<strong>Attack success rate:</strong> {br.most_suspicious_asr*100:.0f}%<br>'
            f'<strong>Assessment:</strong> <span style="color:{overall_col};font-weight:700">{br.overall_assessment}</span>'
            f'</div>', unsafe_allow_html=True)

        if ds is not None:
            st.markdown("#### 🖼️ Trigger Effect Demo")
            demo = next((r for r in ds.records if r.label != br.target_class), ds.records[0])
            try:
                res = demo_trigger_effect(m, demo.filename, br.most_suspicious_position or "bottom-left",
                                          br.trigger_size)
                col_orig, col_trig = st.columns(2)
                for col, key, colour in [(col_orig, "original", "#3fb950"), (col_trig, "triggered", None)]:
                    r = res[key]
                    c = colour or ("#f85149" if r["predicted_label"] == br.target_class else "#d29922")
                    with col:
                        st.image(r["image_array"], caption=f"{key.title()}: {r['predicted_label']} "
                                 f"({r['confidence']*100:.0f}%)", use_container_width=True)
                        st.markdown(f'<span style="color:{c};font-weight:700">→ {r["predicted_label"].upper()} '
                                    f'({r["confidence"]*100:.0f}%)</span>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not run trigger demo: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Provenance page
# ────────────────────────────────────────────────────────────────────────────

def page_provenance():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🔑 INFERENCE PROVENANCE</h1>',
                unsafe_allow_html=True)
    st.caption("Cryptographic SHA-256 hash chain · Fully local · Detects tampering")

    m = _need_model()
    if m is None:
        return
    ds = st.session_state["dataset"]

    src = st.radio("Input image", (["Dataset sample"] if ds else []) + ["Upload"], horizontal=True)
    img_path = None
    if src == "Dataset sample":
        paths = _sample_paths(ds)
        img_path = st.selectbox("Sample", paths, format_func=lambda p: _rel(p, ds.root))
    else:
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if up is not None:
            img_path = _save_upload(up)

    col1, col2, col3, col4 = st.columns(4)
    gen_btn = col1.button("⚡ Generate Inference", type="primary", use_container_width=True, disabled=img_path is None)
    verify_btn = col2.button("🔍 Verify Latest Record", use_container_width=True)
    chain_btn = col3.button("🔗 Verify Entire Chain", use_container_width=True)
    tamper_btn = col4.button("💀 Simulate Tampering", use_container_width=True)
    st.markdown("---")

    if gen_btn:
        with st.spinner("Running inference and creating provenance record …"):
            try:
                record, info = create_inference_record(img_path, m)
                st.session_state["last_inference"] = record
                st.session_state["inference_info"] = info
                st.session_state["tampering_done"] = False
                st.success("Inference record created and appended to chain.")
            except Exception as e:
                st.error(f"Error: {e}")

    if tamper_btn:
        if not _load_chain():
            st.warning("No records in chain. Generate an inference first.")
        elif simulate_tampering(record_idx=-1, field="output", class_names=m.class_names):
            st.session_state["tampering_done"] = True
            st.error("💀 Record tampered! The output field was modified without updating the hash.")

    rec = get_latest_record()
    if rec:
        info = st.session_state.get("inference_info")
        col_img, col_details = st.columns([1, 2])
        with col_img:
            if info:
                st.image(info["image_array"], caption="Input image", use_container_width=True)
                st.markdown(f'<div style="text-align:center;font-size:1.3rem;font-weight:700;color:#58a6ff">'
                            f'{info["predicted_label"].upper()}<br><span style="font-size:0.9rem;color:#8b949e">'
                            f'{info["confidence"]*100:.0f}% confidence</span></div>', unsafe_allow_html=True)
            else:
                st.caption("(Re-generate inference to see image)")
        with col_details:
            st.markdown("**Latest Inference Record**")
            st.markdown(f"**ID:** `{rec.inference_id}`")
            st.markdown(f"**Output:** `{rec.output}` (confidence: {rec.confidence*100:.1f}%)")
            st.markdown(f"**Timestamp:** `{rec.timestamp}`")
            st.markdown(f"**Input hash:** `{rec.input_hash[:24]}…`")
            st.markdown(f"**Model hash:** `{rec.model_hash[:24]}…`")
            st.markdown(f"**Preprocessing hash:** `{rec.preprocessing_hash[:24]}…`")
            st.markdown(f"**Prev hash:** `{rec.previous_record_hash[:24]}…`")
            st.markdown(f"**Record hash:** `{rec.record_hash[:24]}…`")

    if verify_btn and rec:
        vr = verify_inference_record(rec)
        if vr.is_valid:
            st.markdown('<p class="status-verified">✓ RECORD VERIFIED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-tampered">✗ INFERENCE TAMPERED</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="finding-high"><strong>Expected hash:</strong><br>'
                f'<span class="hash-mono">{vr.expected_hash}</span><br><br>'
                f'<strong>Observed hash:</strong><br><span class="hash-mono">{vr.observed_hash}</span><br><br>'
                f'<strong>Broken field:</strong> {vr.broken_field}</div>', unsafe_allow_html=True)

    if chain_btn:
        cr = verify_entire_chain()
        st.session_state["chain_result"] = cr
        if cr.is_intact:
            st.markdown('<p class="status-verified">✓ CHAIN INTACT</p>', unsafe_allow_html=True)
            st.markdown(f"All **{cr.total_records}** records verified ✅")
        else:
            st.markdown('<p class="status-tampered">✗ CHAIN INTEGRITY VIOLATED</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="finding-high"><strong>Broken at record:</strong> {cr.first_broken}<br>'
                f'<strong>Record ID:</strong> {cr.broken_record_id}<br>'
                f'{cr.total_records - cr.valid_records} record(s) invalid out of {cr.total_records}</div>',
                unsafe_allow_html=True)
            for vr in cr.verification_details:
                if not vr.is_valid:
                    st.markdown(
                        f'<div class="finding-high"><strong>Record {vr.record_id[:8]}… FAILED</strong><br>'
                        f'Expected: <span class="hash-mono">{vr.expected_hash}</span><br>'
                        f'Observed: <span class="hash-mono">{vr.observed_hash}</span><br>{vr.error_msg}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Chain History**")
    records = _load_chain()
    if records:
        st.dataframe(pd.DataFrame([{
            "ID": r.inference_id[:8] + "…", "Output": r.output, "Conf": f"{r.confidence*100:.0f}%",
            "Timestamp": r.timestamp[:10], "Hash": r.record_hash[:12] + "…",
        } for r in records[-10:]]), use_container_width=True, hide_index=True)
    else:
        st.info("No inference records yet. Click **⚡ Generate Inference**.")

    col_clr, _ = st.columns([1, 5])
    if col_clr.button("🗑️ Clear Chain", use_container_width=True):
        clear_chain()
        st.session_state["last_inference"] = None
        st.session_state["inference_info"] = None
        st.session_state["tampering_done"] = False
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Assurance Report page
# ────────────────────────────────────────────────────────────────────────────

def page_report():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">📋 ASSURANCE REPORT</h1>', unsafe_allow_html=True)
    rp = st.session_state["report"]
    if not rp:
        st.info("No report generated yet. Go to **Dashboard** and click **▶ Run All Scans**.")
        return

    ds, m = st.session_state["dataset"], st.session_state["model"]
    status_colour = ("#f85149" if "HIGH" in rp.overall_status or "REVIEW" in rp.overall_status
                     else "#d29922" if "CAUTION" in rp.overall_status else "#3fb950")
    st.markdown(
        f'<div class="card"><h2 style="color:{status_colour};font-family:monospace">CV ASSURANCE REPORT</h2>'
        f'<strong>Overall Status:</strong> <span style="color:{status_colour};font-weight:700">{rp.overall_status}</span><br>'
        f'<strong>Dataset:</strong> {ds.name if ds else "—"} <span class="hash-mono">{rp.dataset_path}</span><br>'
        f'<strong>Model:</strong> {m.name if m else "—"} <span class="hash-mono">{rp.model_path}</span></div>',
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Integrity", f"{rp.data_integrity_score}/100")
    col2.metric("Model Integrity", f"{rp.model_integrity_score}/100")
    prov_col = "#3fb950" if rp.provenance_status == "VERIFIED" else "#f85149"
    col3.markdown(f"**Provenance**<br><span style='color:{prov_col};font-weight:700;font-size:1.3rem'>"
                  f"{rp.provenance_status}</span>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 1))
    fig.patch.set_facecolor("#0d1117")
    for ax, (label, score) in zip(axes, [("Data Integrity", rp.data_integrity_score),
                                         ("Model Integrity", rp.model_integrity_score)]):
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
    st.markdown(f'<div class="card"><strong>Recommended Disposition:</strong><br>'
                f'<span style="font-size:1.1rem;color:#58a6ff">{rp.recommended_disposition}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ Scope & Limitations")
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

    st.markdown("---")
    report_data = {
        "overall_status": rp.overall_status,
        "dataset": {"name": ds.name if ds else None, "path": rp.dataset_path,
                    "classes": ds.class_names if ds else None, "images": len(ds) if ds else None},
        "model": {"name": m.name if m else None, "path": rp.model_path, "sha256": m.sha256 if m else None,
                  "arch": m.arch if m else None},
        "reference": st.session_state["reference_name"],
        "data_integrity_score": rp.data_integrity_score,
        "model_integrity_score": rp.model_integrity_score,
        "provenance_status": rp.provenance_status,
        "recommended_disposition": rp.recommended_disposition,
        "findings": [{"severity": f.severity, "category": f.category, "description": f.description,
                      "evidence": f.evidence, "recommendation": f.recommendation} for f in rp.findings],
        "limitations": LIMITATIONS,
        "supported_attacks": SUPPORTED_ATTACKS,
    }
    st.download_button("📥 Download report.json", data=json.dumps(report_data, indent=2),
                       file_name="assurance_report.json", mime="application/json")


# ────────────────────────────────────────────────────────────────────────────
# Page: Explainability (Grad-CAM)
# ────────────────────────────────────────────────────────────────────────────

def _upscale(arr: np.ndarray, size: int = 256) -> np.ndarray:
    """Nearest-neighbour upscale so small images stay crisp on screen."""
    return np.array(Image.fromarray(arr).resize((size, size), Image.NEAREST))


def _heatmap_rgb(heatmap: np.ndarray, size: int = 256) -> np.ndarray:
    hm = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR))
    return (plt.get_cmap("jet")(hm / 255.0)[..., :3] * 255).astype(np.uint8)


def page_explainability():
    st.markdown('<h1 style="color:#58a6ff;font-family:monospace">🔥 GRAD-CAM EXPLAINABILITY</h1>',
                unsafe_allow_html=True)
    st.caption("Shows which image regions drove the active model's prediction")

    m = _need_model()
    if m is None:
        return
    try:
        layers, default_layer = list_target_layers(m)
    except Exception as e:
        st.error(str(e))
        return
    st.info(f"Active model: **{m.name}** (change it in ⚙️ Setup or the sidebar)")
    ds = st.session_state["dataset"]
    h, w = m.input_size

    col_src, col_opts = st.columns([3, 2])
    with col_src:
        source = st.radio("Image source", (["Pick a sample"] if ds else []) + ["Upload an image"], horizontal=True)
        image = None
        if source == "Upload an image":
            up = st.file_uploader(f"Image (resized to {h}×{w})", type=["png", "jpg", "jpeg", "bmp", "webp"])
            if up is not None:
                image = Image.open(up).convert("RGB")
        else:
            choice = st.selectbox("Sample", _sample_paths(ds, 2000), format_func=lambda p: _rel(p, ds.root))
            image = Image.open(choice).convert("RGB")

    with col_opts:
        stamp = st.checkbox("Stamp trigger patch", help="Adds a white square patch, as used by patch backdoors.")
        position = st.selectbox("Trigger position", POSITIONS, index=POSITIONS.index("bottom-left"), disabled=not stamp)
        ts = st.slider("Patch size (px)", 1, max(4, min(h, w) // 3), default_trigger_size(m.input_size),
                       disabled=not stamp)
        explain = st.selectbox("Class to explain", ["Predicted class"] + m.class_names)
        layer_labels = list(layers)
        layer = st.selectbox("Target layer", layer_labels, index=layer_labels.index(default_layer))
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)

    if image is None:
        st.info("Upload an image or pick a sample to see the explanation.")
        return

    clean_pred = None
    if stamp:
        arr = np.array(m.resize(image))
        clean_pred = int(m.predict([arr])[0])
        image = Image.fromarray(apply_trigger(arr, position, ts))

    target = None if explain == "Predicted class" else m.class_names.index(explain)
    try:
        res = explain_image(m, image, target_class=target, layer_name=layers[layer])
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
        return

    st.markdown("---")
    st.markdown(f'<div style="font-size:1.3rem;font-weight:700;color:#58a6ff">Prediction: {res.predicted_label.upper()} '
                f'<span style="font-size:0.95rem;color:#8b949e">({res.confidence*100:.1f}% confidence)</span></div>',
                unsafe_allow_html=True)
    if res.explained_idx != res.predicted_idx:
        st.caption(f"Heatmap explains **{res.explained_label}** "
                   f"({res.probs[res.explained_idx]*100:.1f}% probability), not the predicted class.")

    input_big = _upscale(res.image)
    heat_big = _heatmap_rgb(res.heatmap)
    overlay = ((1 - alpha) * input_big + alpha * heat_big).astype(np.uint8)
    c1, c2, c3 = st.columns(3)
    c1.image(input_big, caption=f"Model input ({h}×{w})", use_container_width=True)
    c2.image(heat_big, caption=f"Grad-CAM for '{res.explained_label}'", use_container_width=True)
    c3.image(overlay, caption="Overlay", use_container_width=True)
    st.caption("Red = regions that most increased the score for the explained class; blue = little influence.")

    top = np.argsort(res.probs)[::-1][:15]           # keep the chart readable for many classes
    fig, ax = plt.subplots(figsize=(8, 2.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.bar([m.class_names[i] for i in top], res.probs[top] * 100,
           color=["#58a6ff" if i == res.predicted_idx else "#30363d" for i in top])
    ax.set_ylabel("Probability (%)", color="#c9d1d9")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if stamp:
        r, c = trigger_origin(h, w, position, ts)
        patch_focus = float(res.heatmap[r:r + ts, c:c + ts].mean())
        overall = float(res.heatmap.mean())
        flipped = clean_pred is not None and res.predicted_idx != clean_pred
        if patch_focus > 2 * overall and flipped:
            st.markdown(
                f'<div class="card finding-high"><strong>Finding:</strong> Attention concentrated on trigger patch<br>'
                f'<strong>Evidence:</strong> Mean activation on patch {patch_focus:.2f} vs {overall:.2f} image-wide; '
                f'prediction flipped from <strong>{m.class_names[clean_pred]}</strong> to '
                f'<strong>{res.predicted_label}</strong>.<br>'
                f'<strong>Interpretation:</strong> Consistent with a patch-triggered backdoor.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card finding-pass"><strong>Trigger patch does not dominate the prediction.</strong><br>'
                f'Mean activation on patch {patch_focus:.2f} vs {overall:.2f} image-wide; '
                f'prediction {"changed" if flipped else "unchanged"} by the patch.</div>', unsafe_allow_html=True)

    st.caption("⚠️ Grad-CAM is a coarse, gradient-based approximation. "
               "It shows correlation with the class score, not proof of causation.")


# ────────────────────────────────────────────────────────────────────────────
# Router
# ────────────────────────────────────────────────────────────────────────────

PAGES = {
    "🏠 Dashboard": page_dashboard,
    "⚙️ Setup": page_setup,
    "📊 Dataset Scanner": page_dataset,
    "🤖 Model Scanner": page_model,
    "🔑 Provenance": page_provenance,
    "🔥 Explainability": page_explainability,
    "📋 Assurance Report": page_report,
}
PAGES[page]()
