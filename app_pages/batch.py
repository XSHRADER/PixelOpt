"""Batch: many images, one set of settings, one ZIP."""

from __future__ import annotations

import statistics

import pandas as pd
import streamlit as st

from app_shared import UPLOAD_TYPES, fingerprint, form_job, target_job
from pixelopt.batch import build_zip, report_csv, run_batch
from pixelopt.enhance import PRESETS
from pixelopt.forms import FORM_PRESETS
from ui_components import hero, metric_strip, thumbnail_url, tone

hero(
    eyebrow="Batch mode",
    title="Many images. One download.",
    subtitle=(
        "Apply one set of settings to a whole folder. Every file is measured, a "
        "broken upload never stops the run, and the ZIP holds exactly the bytes "
        "the report describes."
    ),
    chips=("Target size or form photo", "Per-image auto enhancement", "Damage per image",
           "Failures isolated", "Unique file names", "CSV report", "ZIP export"),
)

files = st.file_uploader("Images", type=UPLOAD_TYPES, accept_multiple_files=True,
                         key="batch_upload")

with st.container(border=True, key="po_card_batch_settings"):
    mode = st.segmented_control(
        "Mode", ["target", "form"], default="target", key="batch_mode",
        format_func={"target": "Target size", "form": "Form photo"}.get,
    ) or "target"
    if mode == "target":
        c1, c2, c3 = st.columns([2, 1, 1])
        target_kb = c1.slider("Target size", 10, 2000, 200, 10, format="%d KB",
                              key="batch_target")
        fmt_choice = c2.selectbox("Encoder", ["Auto", "JPEG", "WebP", "PNG"], key="batch_fmt")
        enhancement = c3.selectbox("Enhancement", ["Auto", *sorted(PRESETS)], key="batch_enh",
                                   help="Auto is decided per image from its own noise level.")
        settings_signature = ("target", target_kb, fmt_choice, enhancement)
    else:
        preset_key = st.selectbox(
            "Form preset", list(FORM_PRESETS), key="batch_form",
            format_func=lambda k: (
                f"{FORM_PRESETS[k].label} · {FORM_PRESETS[k].width}×{FORM_PRESETS[k].height} · "
                f"{FORM_PRESETS[k].min_kb:g}–{FORM_PRESETS[k].max_kb:g} KB"
            ),
        )
        settings_signature = ("form", preset_key)
    measure = st.toggle("Measure damaged area", value=True, key="batch_measure",
                        help="Adds a local-SSIM damage measurement per image, on a "
                             "reduced copy to keep it quick.")

if not files:
    with st.container(horizontal=True, gap="medium"):
        for key, icon, title, body in (
            ("iso", ":material/shield:", "Failures stay isolated",
             "A corrupt or unsupported file is recorded in the report and the rest "
             "of the batch carries on."),
            ("names", ":material/label:", "No silent overwrites",
             "photo.png and photo.jpg would both become photo_pixelopt.jpg, so "
             "every output gets a distinct name."),
            ("zip", ":material/folder_zip:", "The bytes that were measured",
             "The ZIP contains exactly the encodes whose sizes and SSIM appear in "
             "report.csv — nothing is re-encoded on the way out."),
        ):
            with st.container(border=True, key=f"po_card_intro_{key}"):
                st.markdown(f"#### {icon} {title}")
                st.caption(body)
    st.stop()

payloads = [(f.name, f.getvalue()) for f in files]
signature = (settings_signature, measure, tuple((n, fingerprint(r)) for n, r in payloads))

run = st.button(f"Process {len(payloads)} image{'s' if len(payloads) != 1 else ''}",
                type="primary", icon=":material/play_arrow:", key="batch_run")

if run:
    if mode == "target":
        job = target_job(target_kb, None if fmt_choice == "Auto" else fmt_choice.upper(),
                         enhancement, measure)
    else:
        job = form_job(FORM_PRESETS[preset_key], measure)

    progress = st.progress(0.0, text="Starting…")
    ticker = st.empty()

    def on_progress(done: int, total: int, item) -> None:
        progress.progress(done / total, text=f"{done} of {total} · {item.name}")
        ticker.caption(
            (f":material/check_circle: {item.name} → {item.output_bytes / 1024:.1f} KB "
             f"in {item.seconds:.1f}s")
            if item.ok else f":material/error: {item.name} — {item.error}"
        )

    items = run_batch(payloads, job, on_progress=on_progress)
    progress.empty()
    ticker.empty()
    # Session state, so a download click (which reruns the page) does not
    # recompute the whole batch.
    st.session_state["batch_state"] = {
        "signature": signature, "items": items,
        "zip": build_zip(items), "csv": report_csv(items),
    }

state = st.session_state.get("batch_state")
if state is None:
    st.info("Choose settings, then press Process.", icon=":material/touch_app:")
    st.stop()
if state["signature"] != signature:
    st.caption(":material/history: The files or settings have changed since this run — "
               "press Process to update.")

items = state["items"]
ok = [item for item in items if item.ok]
total_in = sum(item.input_bytes for item in ok)
total_out = sum(item.output_bytes for item in ok)
fidelities = [item.fidelity for item in ok if item.fidelity == item.fidelity]
damages = [item.damaged_share for item in ok if item.damaged_share == item.damaged_share]

metric_strip(
    [
        {"label": "Processed", "text": f"{len(ok)} / {len(items)}",
         "hint": f"{len(items) - len(ok)} failed" if len(ok) < len(items) else "no failures",
         "tone": "good" if len(ok) == len(items) else "warn"},
        {"label": "Total in", "value": total_in / 1024 / 1024, "decimals": 2, "suffix": " MB"},
        {"label": "Total out", "value": total_out / 1024 / 1024, "decimals": 2, "suffix": " MB"},
        {"label": "Saved", "value": (1 - total_out / total_in) * 100 if total_in else 0,
         "decimals": 1, "suffix": "%", "hint": "across successful files",
         "tone": "good" if total_out < total_in else "warn"},
        {"label": "Median fidelity",
         "value": statistics.median(fidelities) if fidelities else float("nan"),
         "decimals": 4, "tone": tone(statistics.median(fidelities), 0.98, 0.93) if fidelities else ""},
        {"label": "Worst damaged area",
         "value": max(damages) * 100 if damages else float("nan"), "decimals": 1, "suffix": "%",
         "text": "not measured" if not damages else None,
         "tone": tone(max(damages), 0.02, 0.15, higher_is_better=False) if damages else ""},
    ],
    key="batch_metrics",
)

frame = pd.DataFrame(
    [
        {
            "preview": thumbnail_url(item.data) if item.ok else None,
            "file": item.name,
            "status": "ok" if item.ok else "error",
            "input_kb": item.input_bytes / 1024,
            "output_kb": item.output_bytes / 1024 if item.ok else None,
            "saved": max(0.0, item.saving * 100) if item.ok else None,
            "encoder": item.format or None,
            "size": f"{item.width}×{item.height}" if item.ok else None,
            "fidelity": item.fidelity if item.ok else None,
            "damaged": item.damaged_share * 100
            if item.ok and item.damaged_share == item.damaged_share else None,
            "output": item.output_name or None,
            "notes": item.notes if item.ok else item.error,
        }
        for item in items
    ]
)

st.dataframe(
    frame,
    hide_index=True,
    width="stretch",
    row_height=64,
    column_config={
        "preview": st.column_config.ImageColumn("Preview", width="small"),
        "file": st.column_config.TextColumn("File"),
        "status": st.column_config.TextColumn("Status", width="small"),
        "input_kb": st.column_config.NumberColumn("In", format="%.1f KB"),
        "output_kb": st.column_config.NumberColumn("Out", format="%.1f KB"),
        "saved": st.column_config.ProgressColumn("Saved", format="%.0f%%",
                                                 min_value=0, max_value=100),
        "encoder": st.column_config.TextColumn("Encoder", width="small"),
        "size": st.column_config.TextColumn("Size", width="small"),
        "fidelity": st.column_config.NumberColumn("SSIM", format="%.4f"),
        "damaged": st.column_config.NumberColumn("Damaged", format="%.1f%%"),
        "output": st.column_config.TextColumn("Output name"),
        "notes": st.column_config.TextColumn("Notes", width="large"),
    },
)

with st.container(horizontal=True, gap="small"):
    st.download_button(
        f"Download ZIP · {len(state['zip']) / 1024 / 1024:.2f} MB", data=state["zip"],
        file_name="pixelopt_batch.zip", mime="application/zip", type="primary",
        icon=":material/folder_zip:", disabled=not ok,
    )
    st.download_button("Report (CSV)", data=state["csv"], file_name="pixelopt_report.csv",
                       mime="text/csv", icon=":material/table:")
    if st.button("Clear results", icon=":material/delete_sweep:", key="batch_clear"):
        st.session_state.pop("batch_state", None)
        st.rerun()
