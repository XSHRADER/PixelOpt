"""Custom UI for the Streamlit app.

Motion follows the same restraint as the design it borrows from: interactions
answer in 100-160 ms on an ease-out curve, ambient motion is slow enough to
ignore, and nothing bounces. Every animation switches off under
prefers-reduced-motion.

Where the motion lives matters. Streamlit redraws the page on every
interaction, so smooth transitions cannot run *between* reruns in native
elements. Anything that must animate from an old value to a new one -- the
count-up metrics, the range meter, the comparison modes -- is a Components v2
element that keeps its previous state across updates. The page header is
static HTML and does not move.
"""

from __future__ import annotations

import base64
import html
import io
import math
import re
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

PREVIEW_MAX_SIDE = 2400
PREVIEW_QUALITY = 92
HEATMAP_MAX_SIDE = 1400


# --------------------------------------------------------------- data URLs


def image_data_url(picture: Image.Image, max_side: int = PREVIEW_MAX_SIDE) -> str:
    """WebP at 92 is visually lossless for the reference layer."""
    copy = picture.convert("RGB")
    if max(copy.size) > max_side:
        copy.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    copy.save(buffer, format="WEBP", quality=PREVIEW_QUALITY, method=4)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def bytes_data_url(raw: bytes, mime: str) -> str:
    """The encoded output verbatim, so what is inspected is what is downloaded."""
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def rgba_data_url(array: np.ndarray, max_side: int = HEATMAP_MAX_SIDE) -> str:
    picture = Image.fromarray(array.astype(np.uint8), mode="RGBA")
    if max(picture.size) > max_side:
        picture.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def thumbnail_url(raw: bytes, side: int = 88) -> Optional[str]:
    try:
        picture = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None
    picture.thumbnail((side, side), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ----------------------------------------------------------------- palette


# Static HTML (st.html) cannot read Streamlit's theme variables, so the few
# colours it needs are repeated from .streamlit/config.toml. The theme has no
# light variant, so there is one set.
_PALETTE: Dict[str, str] = {
    "BG": "#1f2023", "CARD": "#2a2b2f", "BORDER": "#3a3c41", "TEXT": "#e8e8ea",
    "MUTED": "#9a9ca3", "ACCENT": "#5cc8b8", "ACCENT_SOFT": "rgba(92,200,184,0.35)",
}


def _fill(template: str) -> str:
    for name, value in _PALETTE.items():
        template = template.replace(f"__{name}__", value)
    return template


# -------------------------------------------------------------- app styling

_APP_CSS = """
<style>
/* Colour and type come from .streamlit/config.toml. This covers only what
   the theme cannot express, through stable hooks: data-testids for the
   chrome, and st-key-* classes on containers this app creates. */
header[data-testid="stHeader"] {
  background: __BG__ !important;
  border-bottom: 1px solid __BORDER__;
}
[class*="st-key-po_card"] > div { transition: border-color .16s ease; }
[class*="st-key-po_card"]:hover > div { border-color: __ACCENT_SOFT__; }
[data-testid="stFileUploaderDropzone"] { transition: border-color .16s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: __ACCENT__; }
/* Figures line up in columns when they share a monospaced face. */
[class*="st-key-po_numbers"] [data-testid="stTable"] td {
  font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums;
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def app_style() -> None:
    st.html(_fill(_APP_CSS))


# ------------------------------------------------------------------ header

_HEADER_CSS = """
<style>
.po-head { padding: 22px 0 8px; font-family: Inter, -apple-system, "Segoe UI", sans-serif; }
.po-eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 500;
  color: __MUTED__; padding: 3px 10px 3px 8px; border: 1px solid __BORDER__; border-radius: 6px;
  background: __CARD__; }
.po-dot { width: 6px; height: 6px; border-radius: 50%; background: __ACCENT__; }
.po-title { margin: 14px 0 8px; padding: 0; font-size: clamp(28px, 3.2vw, 40px); line-height: 1.1;
  letter-spacing: -0.02em; font-weight: 600; color: __TEXT__; max-width: 24ch; }
.po-sub { max-width: 66ch; margin: 0; font-size: 15px; line-height: 1.6; color: __MUTED__; }
</style>
"""


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """The page introduction, shown only until something is uploaded."""
    st.html(
        _fill(_HEADER_CSS)
        + '<section class="po-head">'
        + f'<div class="po-eyebrow"><span class="po-dot"></span>{html.escape(eyebrow)}</div>'
        + f'<h1 class="po-title">{html.escape(title)}</h1>'
        + f'<p class="po-sub">{html.escape(subtitle)}</p>'
        + "</section>"
    )


def intro_cards(cards: Sequence[Tuple[str, str, str, str]]) -> None:
    """A row of bordered explainer cards, each (key, icon, title, body)."""
    with st.container(horizontal=True, gap="medium"):
        for key, icon, title, body in cards:
            with st.container(border=True, key=f"po_card_intro_{key}"):
                st.markdown(f"#### {icon} {title}")
                st.caption(body)


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$:])")


def file_line(name: str, facts: str) -> None:
    """One line naming what is loaded, in place of the page header."""
    # File names are user input: escape them, or photo_1_final.jpg renders
    # half in italics.
    escaped = _MARKDOWN_SPECIAL.sub(lambda match: "\\" + match.group(1), name)
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.markdown(f"**{escaped}**", width="content")
        st.caption(facts, width="content")


def workbench(key: str) -> Tuple["DeltaGenerator", "DeltaGenerator"]:
    """Controls on the left, the result on the right.

    The point is cause and effect side by side: change a setting and its
    result updates next to it rather than a scroll away. Returns the bordered
    control panel and the result column.
    """
    left, right = st.columns([1, 2.4], gap="medium")
    return left.container(border=True, key=f"po_panel_{key}"), right


# ---------------------------------------------------------- metric strip

_METRICS_HTML = """<div class="ms" id="ms"></div>"""

_METRICS_CSS = """
.ms { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px;
  font-family: var(--st-font, Inter, sans-serif); }
.ms-card { padding: 11px 13px 10px; background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color); border-radius: var(--st-base-radius, 6px);
  transition: border-color .16s ease; }
.ms-card:hover { border-color: color-mix(in srgb, var(--st-primary-color) 55%, var(--st-border-color)); }
.ms-label { display: flex; align-items: center; gap: 7px; font-size: 12px;
  color: color-mix(in srgb, var(--st-text-color) 62%, transparent); }
.ms-dot { width: 6px; height: 6px; border-radius: 50%; flex: none;
  background: color-mix(in srgb, var(--st-text-color) 35%, transparent);
  transition: background-color .3s ease; }
.ms-good { background: var(--st-green-color); }
.ms-warn { background: var(--st-yellow-color); }
.ms-bad { background: var(--st-red-color); }
.ms-value { margin-top: 5px; font-family: var(--st-code-font, "JetBrains Mono", monospace);
  font-size: 20px; line-height: 1.2; font-weight: 500; letter-spacing: -0.01em;
  font-variant-numeric: tabular-nums; color: var(--st-text-color);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ms-hint { margin-top: 3px; font-size: 11.5px; min-height: 15px;
  color: color-mix(in srgb, var(--st-text-color) 60%, transparent);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (prefers-reduced-motion: reduce) { .ms-card, .ms-dot { transition: none; } }
"""

_METRICS_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector("#ms");
  if (!root) return;
  const items = (data && data.items) || [];
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Rebuild the cards only when the set of metrics changes. Otherwise the
  // numbers morph in place from their previous values, which is the point:
  // move the budget slider and you watch the size and quality follow it.
  const signature = items.map((it) => it.label).join("|");
  if (parentElement.__msSignature !== signature) {
    root.innerHTML = items.map((_, i) =>
      '<div class="ms-card">' +
      '<div class="ms-label"><span class="ms-dot"></span><span class="ms-text"></span></div>' +
      '<div class="ms-value"></div><div class="ms-hint"></div></div>'
    ).join("");
    parentElement.__msSignature = signature;
    parentElement.__msPrevious = {};
  }
  const previous = parentElement.__msPrevious || {};
  const cards = root.querySelectorAll(".ms-card");

  items.forEach((it, i) => {
    const card = cards[i];
    if (!card) return;
    card.querySelector(".ms-text").textContent = it.label;
    card.querySelector(".ms-dot").className = "ms-dot" + (it.tone ? " ms-" + it.tone : "");
    card.querySelector(".ms-hint").textContent = it.hint || "";
    const el = card.querySelector(".ms-value");

    if (typeof it.value !== "number" || !isFinite(it.value)) {
      el.textContent = it.text != null ? it.text : "\\u2013";
      return;
    }
    const decimals = it.decimals || 0;
    const format = (v) => (it.prefix || "") +
      v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) +
      (it.suffix || "");
    const from = typeof previous[it.label] === "number" ? previous[it.label] : 0;
    const to = it.value;
    cancelAnimationFrame(el.__raf);
    if (reduce || Math.abs(to - from) < 1e-12) { el.textContent = format(to); return; }
    const start = performance.now();
    const duration = 950;
    const step = (now) => {
      const k = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      el.textContent = format(from + (to - from) * eased);
      if (k < 1) el.__raf = requestAnimationFrame(step);
    };
    el.__raf = requestAnimationFrame(step);
  });

  const next = {};
  for (const it of items) if (typeof it.value === "number") next[it.label] = it.value;
  parentElement.__msPrevious = next;
}
"""

_METRICS = st.components.v2.component(
    "pixelopt_metrics", html=_METRICS_HTML, css=_METRICS_CSS, js=_METRICS_JS,
    isolate_styles=True,
)


def metric_strip(items: Sequence[Dict[str, object]], *, key: str) -> None:
    """Metric cards whose numbers count up, then morph between reruns.

    Each item: label, and either `value` (number, with optional decimals,
    prefix, suffix) or `text`; optional `hint` and `tone` (good/warn/bad).
    """
    clean = []
    for item in items:
        entry = {k: v for k, v in dict(item).items() if v is not None}
        if "value" in entry:
            value = float(entry["value"])
            # NaN is not valid JSON, and a missing measurement should read as
            # its `text` (or a dash), not as a number.
            if math.isfinite(value):
                entry["value"] = value
            else:
                entry.pop("value")
        clean.append(entry)
    _METRICS(key=key, data={"items": clean})


def tone(value: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    if value != value:  # NaN
        return ""
    if higher_is_better:
        return "good" if value >= good else "warn" if value >= warn else "bad"
    return "good" if value <= good else "warn" if value <= warn else "bad"


# ------------------------------------------------------------ range meter

_RANGE_HTML = """
<div class="rm">
  <div class="rm-head">
    <span class="rm-title">File size against the allowed range</span>
    <span class="rm-status" id="status"></span>
  </div>
  <div class="rm-track">
    <div class="rm-band" id="band"><span class="rm-band-label" id="bandLabel"></span></div>
    <div class="rm-pad" id="pad"></div>
    <div class="rm-ghost" id="ghost"></div>
    <div class="rm-marker" id="marker"><div class="rm-bubble" id="bubble"></div><div class="rm-pin"></div></div>
  </div>
  <div class="rm-scale"><span>0 KB</span><span id="scaleMax"></span></div>
</div>
"""

_RANGE_CSS = """
.rm { font-family: var(--st-font, Inter, sans-serif); color: var(--st-text-color);
  padding: 14px 16px 10px; border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius, 6px); background: var(--st-secondary-background-color); }
.rm-head { display: flex; justify-content: space-between; align-items: center; gap: 12px;
  font-size: 12.5px; margin-bottom: 38px; }
.rm-title { color: color-mix(in srgb, var(--st-text-color) 62%, transparent); }
.rm-status { font-weight: 600; padding: 3px 10px; border-radius: 4px; font-size: 12px;
  transition: background-color .3s ease, color .3s ease; }
.rm-status.ok { color: var(--st-green-color); background: color-mix(in srgb, var(--st-green-color) 14%, transparent); }
.rm-status.pad { color: var(--st-yellow-color); background: color-mix(in srgb, var(--st-yellow-color) 14%, transparent); }
.rm-status.over { color: var(--st-red-color); background: color-mix(in srgb, var(--st-red-color) 14%, transparent); }
.rm-track { position: relative; height: 10px; border-radius: 999px;
  background: color-mix(in srgb, var(--st-text-color) 8%, transparent); }
.rm-band { position: absolute; top: 0; bottom: 0; border-radius: 999px;
  background: color-mix(in srgb, var(--st-green-color) 45%, transparent);
  transition: left .9s cubic-bezier(.22,1,.36,1), width .9s cubic-bezier(.22,1,.36,1); }
.rm-band-label { position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
  font-size: 11px; white-space: nowrap; color: var(--st-green-color); }
.rm-pad { position: absolute; top: 3px; height: 4px; border-radius: 999px; opacity: 0;
  background: repeating-linear-gradient(90deg, var(--st-yellow-color) 0 5px, transparent 5px 9px);
  transition: left .9s cubic-bezier(.22,1,.36,1), width .9s cubic-bezier(.22,1,.36,1), opacity .4s ease .5s; }
.rm-pad.on { opacity: 1; }
.rm-ghost { position: absolute; top: -4px; width: 2px; height: 18px; margin-left: -1px; opacity: 0;
  background: color-mix(in srgb, var(--st-text-color) 45%, transparent);
  transition: left .9s cubic-bezier(.22,1,.36,1), opacity .4s ease; }
.rm-ghost.on { opacity: 1; }
.rm-marker { position: absolute; top: -7px; width: 0; left: 0;
  transition: left .9s cubic-bezier(.22,1,.36,1); }
.rm-pin { position: absolute; left: -8px; top: 0; width: 16px; height: 24px;
  border-radius: 8px; background: var(--st-text-color);
  box-shadow: 0 0 0 3px var(--st-secondary-background-color); }
.rm-bubble { position: absolute; bottom: 30px; left: 0; transform: translateX(-50%);
  font-family: var(--st-code-font, monospace); font-size: 12px; font-weight: 500; white-space: nowrap;
  padding: 3px 8px; border-radius: 4px; color: var(--st-background-color); background: var(--st-text-color);
  font-variant-numeric: tabular-nums; }
.rm-scale { display: flex; justify-content: space-between; margin-top: 30px; font-size: 11px;
  color: color-mix(in srgb, var(--st-text-color) 55%, transparent); }
@media (prefers-reduced-motion: reduce) {
  .rm-band, .rm-pad, .rm-ghost, .rm-marker { transition: none; } }
"""

_RANGE_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const q = (s) => parentElement.querySelector(s);
  const marker = q("#marker"), band = q("#band"), pad = q("#pad"), ghost = q("#ghost");
  if (!marker || !band) return;
  const d = data || {};
  const low = d.low || 0, high = d.high || 1, actual = d.actual || 0;
  const natural = d.natural || actual;
  const scale = Math.max(high * 1.3, actual * 1.12, natural * 1.12, 1);
  const pct = (v) => Math.max(0, Math.min(100, (v / scale) * 100)) + "%";
  const kb = (b) => (b / 1024).toFixed(1) + " KB";

  // First paint starts everything at zero, so the marker glides in to where
  // the file landed. Later updates animate from wherever it currently is.
  if (!parentElement.__rmMounted) {
    marker.style.left = "0%"; band.style.left = "0%"; band.style.width = "0%";
    void marker.offsetWidth;
    parentElement.__rmMounted = true;
  }
  requestAnimationFrame(() => {
    band.style.left = pct(low);
    band.style.width = "calc(" + pct(high) + " - " + pct(low) + ")";
    marker.style.left = pct(actual);
    const padded = actual > natural + 0.5;
    pad.classList.toggle("on", padded);
    ghost.classList.toggle("on", padded);
    if (padded) {
      pad.style.left = pct(natural);
      pad.style.width = "calc(" + pct(actual) + " - " + pct(natural) + ")";
      ghost.style.left = pct(natural);
    }
  });

  q("#bubble").textContent = kb(actual);
  q("#bandLabel").textContent = "allowed " + kb(low) + " \\u2013 " + kb(high);
  q("#scaleMax").textContent = kb(scale);
  const status = q("#status");
  if (actual > high) { status.className = "rm-status over"; status.textContent = "Over the limit"; }
  else if (actual < low) { status.className = "rm-status over"; status.textContent = "Under the minimum"; }
  else if (actual > natural + 0.5) { status.className = "rm-status pad"; status.textContent = "In range \\u00b7 padded"; }
  else { status.className = "rm-status ok"; status.textContent = "In range"; }
}
"""

_RANGE = st.components.v2.component(
    "pixelopt_range", html=_RANGE_HTML, css=_RANGE_CSS, js=_RANGE_JS, isolate_styles=True,
)


def range_meter(low_bytes: int, high_bytes: int, actual_bytes: int,
                natural_bytes: int, *, key: str) -> None:
    """Animated bar: the allowed band, where the file landed, any padding."""
    _RANGE(key=key, data={"low": int(low_bytes), "high": int(high_bytes),
                          "actual": int(actual_bytes), "natural": int(natural_bytes)})


# ------------------------------------------------------- comparison viewer

_COMPARE_HTML = """
<div class="cmp mode-slide" id="root">
  <div class="cmp-view" id="view">
    <div class="cmp-layer" id="layerBefore"><div class="cmp-pan" id="panBefore">
      <img id="imgBefore" alt="reference" draggable="false">
    </div></div>
    <div class="cmp-layer" id="layerAfter"><div class="cmp-pan" id="panAfter">
      <img id="imgAfter" alt="output" draggable="false">
      <img id="imgHeat" class="cmp-heat" alt="" draggable="false">
      <div class="cmp-worst" id="worst"></div>
    </div></div>
    <div class="cmp-tag cmp-tag-l" id="tagBefore"></div>
    <div class="cmp-tag cmp-tag-r" id="tagAfter"></div>
    <div class="cmp-handle" id="handle"><div class="cmp-grip">
      <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
        <path d="M9.5 7 5 12l4.5 5M14.5 7 19 12l-4.5 5" fill="none" stroke="currentColor"
              stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div></div>
  </div>
  <div class="cmp-bar">
    <div class="cmp-seg" id="modes">
      <button type="button" data-mode="slide">Slide</button>
      <button type="button" data-mode="flicker">Flicker</button>
      <button type="button" data-mode="heat">Heatmap</button>
    </div>
    <div class="cmp-legend"><span>0</span><i></i><span>0.25+ loss</span></div>
    <div class="cmp-spacer"></div>
    <div class="cmp-zoom" id="zoomLabel">fit</div>
    <div class="cmp-actions">
      <button type="button" data-zoom="fit">Fit</button>
      <button type="button" data-zoom="1">1:1</button>
      <button type="button" data-zoom="2">2:1</button>
      <button type="button" data-zoom="4">4:1</button>
    </div>
  </div>
  <div class="cmp-hint" id="hint"></div>
</div>
"""

_COMPARE_CSS = """
.cmp { width: 100%; font-family: var(--st-font, Inter, sans-serif); color: var(--st-text-color); }
.cmp-view { position: relative; aspect-ratio: var(--ar, 1.5); overflow: hidden;
  /* Height is capped by narrowing the box, never by letterboxing inside it:
     the heatmap and worst-region box are placed in percentages of the box,
     so it must keep the image's exact aspect ratio. */
  width: min(100%, calc(var(--cap, 100000px) * var(--ar, 1.5))); margin: 0 auto;
  cursor: grab; touch-action: none; border-radius: var(--st-base-radius, 6px);
  border: 1px solid var(--st-border-color); background-color: var(--st-secondary-background-color);
  background-image:
    linear-gradient(45deg, rgba(128,128,128,.08) 25%, transparent 25% 75%, rgba(128,128,128,.08) 75%),
    linear-gradient(45deg, rgba(128,128,128,.08) 25%, transparent 25% 75%, rgba(128,128,128,.08) 75%);
  background-size: 22px 22px; background-position: 0 0, 11px 11px; }
.cmp-view.is-panning { cursor: grabbing; }
.cmp-view.is-sliding { cursor: ew-resize; }
.cmp-layer { position: absolute; inset: 0; }
/* The divider is a clip on this untransformed wrapper, so it stays put in
   viewport space however far the images are panned or zoomed. */
#layerAfter { clip-path: inset(0 0 0 var(--split, 50%)); }
.cmp.mode-flicker #layerAfter, .cmp.mode-heat #layerAfter { clip-path: inset(0 0 0 0); }
.cmp.mode-flicker.show-before #layerAfter { clip-path: inset(0 0 0 100%); }
.cmp.animating:not(.mode-flicker) #layerAfter { transition: clip-path .5s cubic-bezier(.22,1,.36,1); }
.cmp-pan { position: absolute; inset: 0; transform-origin: 0 0; will-change: transform; }
.cmp-pan img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;
  image-rendering: var(--smoothing, auto); user-select: none; -webkit-user-drag: none; }
.cmp-heat { opacity: 0; pointer-events: none; transition: opacity .5s cubic-bezier(.22,1,.36,1); }
.cmp.mode-heat .cmp-heat { opacity: .92; }
.cmp-worst { position: absolute; display: none; pointer-events: none; border-radius: 6px;
  border: 1.5px solid #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.55);
  animation: cmp-pulse 1.9s ease-in-out infinite; }
.cmp.mode-heat .cmp-worst.on { display: block; }

.cmp-handle { position: absolute; top: 0; bottom: 0; left: var(--split, 50%); width: 2px;
  margin-left: -1px; background: var(--st-primary-color); pointer-events: none;
  box-shadow: 0 0 0 1px rgba(0,0,0,.35);
  transition: opacity .25s ease; }
.cmp.mode-flicker .cmp-handle, .cmp.mode-heat .cmp-handle { opacity: 0; }
.cmp-grip { position: absolute; top: 50%; left: 50%; width: 30px; height: 30px;
  transform: translate(-50%, -50%); display: grid; place-items: center; border-radius: 50%;
  background: var(--st-primary-color); color: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.4);
  transition: transform .18s cubic-bezier(.25,.46,.45,.94); }
.cmp-view:hover .cmp-grip { transform: translate(-50%, -50%) scale(1.1); }
.cmp-view.is-sliding .cmp-grip { transform: translate(-50%, -50%) scale(.94); }

.cmp-tag { position: absolute; top: 10px; padding: 3px 10px; border-radius: 4px;
  font-size: 11.5px; font-weight: 600; color: #fff; background: rgba(31,32,35,.8);
  border: 1px solid rgba(255,255,255,.1); pointer-events: none;
  opacity: 0; transform: translateY(-5px);
  transition: opacity .3s ease .08s, transform .3s ease .08s, background-color .15s ease; }
.cmp.ready .cmp-tag { opacity: 1; transform: none; }
.cmp.mode-heat .cmp-tag-l { opacity: 0; }
.cmp-tag.active { background: var(--st-primary-color); border-color: transparent; }
.cmp-tag-l { left: 10px; }
.cmp-tag-r { right: 10px; }

.cmp-bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 10px; font-size: 12px; }
.cmp-spacer { flex: 1; }
.cmp-seg { display: inline-flex; gap: 2px; padding: 3px; border-radius: var(--st-button-radius, 6px);
  background: var(--st-secondary-background-color); border: 1px solid var(--st-border-color); }
.cmp-seg button, .cmp-actions button { font: inherit; font-size: 12px; font-weight: 500; cursor: pointer;
  color: color-mix(in srgb, var(--st-text-color) 70%, transparent); background: transparent;
  border: 1px solid transparent; border-radius: 4px; padding: 4px 12px;
  transition: color .16s cubic-bezier(.25,.46,.45,.94), background-color .16s cubic-bezier(.25,.46,.45,.94),
              border-color .16s ease, box-shadow .16s ease; }
.cmp-seg button:hover, .cmp-actions button:hover { color: var(--st-text-color); }
.cmp-seg button[aria-pressed="true"] { color: var(--st-text-color); background: var(--st-background-color);
  border-color: var(--st-border-color); box-shadow: 0 1px 3px rgba(0,0,0,.35); }
.cmp-seg button:disabled { opacity: .35; cursor: not-allowed; }
.cmp-actions { display: flex; gap: 4px; }
.cmp-actions button { border-color: var(--st-border-color); }
.cmp-actions button[aria-pressed="true"] { color: #fff; background: var(--st-primary-color);
  border-color: var(--st-primary-color); }
.cmp-zoom { min-width: 46px; text-align: center; font-family: var(--st-code-font, monospace);
  font-weight: 500; font-variant-numeric: tabular-nums;
  padding: 4px 8px; border-radius: 4px; background: var(--st-secondary-background-color); }
.cmp-legend { display: none; align-items: center; gap: 6px; font-size: 11px;
  color: color-mix(in srgb, var(--st-text-color) 60%, transparent); }
.cmp.mode-heat .cmp-legend { display: inline-flex; animation: cmp-fade .4s ease both; }
.cmp-legend i { width: 90px; height: 6px; border-radius: 999px;
  background: linear-gradient(90deg, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4); }
.cmp-hint { margin-top: 7px; font-size: 11.5px; min-height: 16px;
  color: color-mix(in srgb, var(--st-text-color) 45%, transparent); animation: cmp-fade .4s ease both; }

@keyframes cmp-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes cmp-pulse { 50% { box-shadow: 0 0 0 6px rgba(255,255,255,.16), 0 0 0 1px rgba(0,0,0,.55); } }
@media (prefers-reduced-motion: reduce) {
  .cmp * { animation: none !important; transition: none !important; } }
"""

_COMPARE_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const q = (s) => parentElement.querySelector(s);
  const root = q("#root"), view = q("#view");
  const panes = [q("#panBefore"), q("#panAfter")];
  const imgBefore = q("#imgBefore"), imgAfter = q("#imgAfter"), imgHeat = q("#imgHeat");
  const worst = q("#worst"), zoomLabel = q("#zoomLabel"), hint = q("#hint");
  const tagBefore = q("#tagBefore"), tagAfter = q("#tagAfter");
  if (!root || !view || !imgBefore || !imgAfter) return;

  // This runs again on every data update, so stop the previous run's timers.
  clearInterval(parentElement.__cmpFlicker);
  cancelAnimationFrame(parentElement.__cmpSweep);
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const d = data || {};
  if (imgBefore.getAttribute("src") !== d.before) imgBefore.setAttribute("src", d.before || "");
  if (imgAfter.getAttribute("src") !== d.after) imgAfter.setAttribute("src", d.after || "");
  const hasHeat = typeof d.heatmap === "string" && d.heatmap.length > 0;
  if (hasHeat) {
    if (imgHeat.getAttribute("src") !== d.heatmap) imgHeat.setAttribute("src", d.heatmap);
  } else {
    imgHeat.removeAttribute("src");
  }
  tagBefore.textContent = d.labelBefore || "before";
  tagAfter.textContent = d.labelAfter || "after";
  if (d.aspect) view.style.setProperty("--ar", String(d.aspect));
  if (d.reserve > 0) {
    view.style.setProperty("--cap", "max(280px, calc(100vh - " + d.reserve + "px))");
  } else {
    view.style.removeProperty("--cap");
  }
  if (Array.isArray(d.worst) && d.worst.length === 4) {
    const [x0, y0, x1, y1] = d.worst;
    worst.style.left = x0 * 100 + "%";
    worst.style.top = y0 * 100 + "%";
    worst.style.width = (x1 - x0) * 100 + "%";
    worst.style.height = (y1 - y0) * 100 + "%";
    worst.classList.add("on");
  } else {
    worst.classList.remove("on");
  }

  // Persisted across reruns so changing a setting does not throw away the
  // spot being inspected.
  let split = typeof d.split === "number" ? d.split : 50;
  let zoom = typeof d.zoom === "number" ? d.zoom : 1;
  let tx = 0, ty = 0;
  let fitting = d.fitting !== false;
  let mode = ["slide", "flicker", "heat"].includes(d.mode) ? d.mode : "slide";
  if (mode === "heat" && !hasHeat) mode = "slide";
  let seen = d.seen === true;

  const persist = () => {
    clearTimeout(parentElement.__cmpSave);
    parentElement.__cmpSave = setTimeout(() => {
      setStateValue("view", { split, zoom, fitting, mode, seen });
    }, 280);  // debounced: a rerun per wheel tick would be unusable
  };

  const setSplit = (v) => {
    split = Math.min(100, Math.max(0, v));
    view.style.setProperty("--split", split + "%");
  };

  const apply = () => {
    // One transform for both layers -- they cannot drift apart.
    const t = "translate(" + tx + "px, " + ty + "px) scale(" + zoom + ")";
    for (const pane of panes) if (pane) pane.style.transform = t;
    // Nearest-neighbour past 2x so blocking and ringing stay visible.
    view.style.setProperty("--smoothing", zoom >= 2 ? "pixelated" : "auto");
    zoomLabel.textContent = fitting ? "fit" : zoom.toFixed(1) + "x";
    for (const b of parentElement.querySelectorAll(".cmp-actions button")) {
      const z = b.dataset.zoom;
      b.setAttribute("aria-pressed",
        String(z === "fit" ? fitting : !fitting && Math.abs(zoom - Number(z)) < 0.01));
    }
  };

  const clamp = () => {
    const w = view.clientWidth, h = view.clientHeight;
    tx = Math.min(0, Math.max(Math.min(0, w - w * zoom), tx));
    ty = Math.min(0, Math.max(Math.min(0, h - h * zoom), ty));
  };

  const zoomAbout = (factor, cx, cy) => {
    const next = Math.min(12, Math.max(1, zoom * factor));
    if (Math.abs(next - zoom) < 1e-6) return;
    const k = next / zoom;
    tx = cx - (cx - tx) * k;
    ty = cy - (cy - ty) * k;
    zoom = next;
    fitting = Math.abs(zoom - 1) < 1e-6;
    if (fitting) { tx = 0; ty = 0; }
    clamp(); apply(); persist();
  };

  const cancelSweep = () => {
    if (parentElement.__cmpSweep) {
      cancelAnimationFrame(parentElement.__cmpSweep);
      parentElement.__cmpSweep = 0;
      seen = true;
    }
  };

  // First look only: glide the divider across and back, so it is obvious it
  // can be dragged. Never replays once the viewer has touched anything.
  const sweep = () => {
    if (reduce || seen || mode !== "slide") return;
    const frames = [[0, 50], [0.34, 76], [0.7, 26], [1, 50]];
    const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const start = performance.now(), duration = 1800;
    const tick = (now) => {
      const k = Math.min(1, (now - start) / duration);
      let i = 0;
      while (i < frames.length - 2 && k > frames[i + 1][0]) i++;
      const [k0, v0] = frames[i], [k1, v1] = frames[i + 1];
      setSplit(v0 + (v1 - v0) * ease(Math.min(1, Math.max(0, (k - k0) / (k1 - k0)))));
      if (k < 1) {
        parentElement.__cmpSweep = requestAnimationFrame(tick);
      } else {
        parentElement.__cmpSweep = 0;
        seen = true;
        persist();
      }
    };
    parentElement.__cmpSweep = requestAnimationFrame(tick);
  };

  const HINTS = {
    slide: "Drag the divider \\u00b7 scroll to zoom \\u00b7 drag to pan \\u00b7 double-click to reset",
    flicker: "Blinking between reference and output \\u2014 the eye catches a change far faster than side by side",
    heat: "Brighter means more local quality lost \\u00b7 the pulsing box marks the worst region",
  };

  const setMode = (next, animate) => {
    mode = next;
    for (const m of ["slide", "flicker", "heat"]) root.classList.toggle("mode-" + m, m === mode);
    root.classList.remove("show-before");
    if (animate && !reduce) {
      root.classList.add("animating");
      clearTimeout(parentElement.__cmpAnimate);
      parentElement.__cmpAnimate = setTimeout(() => root.classList.remove("animating"), 560);
    }
    for (const b of parentElement.querySelectorAll("#modes button")) {
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
      if (b.dataset.mode === "heat") b.disabled = !hasHeat;
    }
    tagBefore.classList.remove("active");
    tagAfter.classList.remove("active");
    clearInterval(parentElement.__cmpFlicker);
    if (mode === "flicker") {
      let showBefore = false;
      tagAfter.classList.add("active");
      parentElement.__cmpFlicker = setInterval(() => {
        showBefore = !showBefore;
        root.classList.toggle("show-before", showBefore);
        tagBefore.classList.toggle("active", showBefore);
        tagAfter.classList.toggle("active", !showBefore);
      }, 560);
    }
    hint.textContent = HINTS[mode];
    hint.style.animation = "none";
    void hint.offsetWidth;
    hint.style.animation = "";
  };

  view.onwheel = (e) => {
    e.preventDefault();
    cancelSweep();
    const r = view.getBoundingClientRect();
    zoomAbout(e.deltaY < 0 ? 1.16 : 1 / 1.16, e.clientX - r.left, e.clientY - r.top);
  };

  let drag = null, startX = 0, startY = 0, baseTx = 0, baseTy = 0;
  const HANDLE_GRAB_PX = 26;

  view.onpointerdown = (e) => {
    cancelSweep();
    const r = view.getBoundingClientRect();
    const x = e.clientX - r.left;
    const nearHandle = Math.abs(x - (split / 100) * r.width) <= HANDLE_GRAB_PX;
    // At fit there is nothing to pan, so anywhere in slide mode moves the divider.
    drag = mode === "slide" && (nearHandle || zoom <= 1.0001) ? "slide" : "pan";
    view.classList.toggle("is-sliding", drag === "slide");
    view.classList.toggle("is-panning", drag === "pan");
    startX = e.clientX; startY = e.clientY; baseTx = tx; baseTy = ty;
    view.setPointerCapture(e.pointerId);
    if (drag === "slide") setSplit((x / r.width) * 100);
  };

  view.onpointermove = (e) => {
    if (!drag) return;
    const r = view.getBoundingClientRect();
    if (drag === "slide") {
      setSplit(((e.clientX - r.left) / r.width) * 100);
    } else {
      tx = baseTx + (e.clientX - startX);
      ty = baseTy + (e.clientY - startY);
      if (zoom > 1.0001) fitting = false;
      clamp(); apply();
    }
  };

  const end = (e) => {
    if (!drag) return;
    drag = null;
    view.classList.remove("is-sliding", "is-panning");
    if (e && e.pointerId != null && view.hasPointerCapture(e.pointerId)) {
      view.releasePointerCapture(e.pointerId);
    }
    persist();
  };
  view.onpointerup = end;
  view.onpointercancel = end;
  view.ondblclick = () => { zoom = 1; tx = 0; ty = 0; fitting = true; apply(); persist(); };

  for (const b of parentElement.querySelectorAll(".cmp-actions button")) {
    b.onclick = () => {
      cancelSweep();
      const z = b.dataset.zoom;
      if (z === "fit") { zoom = 1; tx = 0; ty = 0; fitting = true; }
      else { zoom = Number(z); fitting = false; tx = 0; ty = 0; clamp(); }
      apply(); persist();
    };
  }
  for (const b of parentElement.querySelectorAll("#modes button")) {
    b.onclick = () => {
      if (b.disabled || b.dataset.mode === mode) return;
      cancelSweep();
      setMode(b.dataset.mode, true);
      persist();
    };
  }

  setSplit(split);
  setMode(mode, false);
  apply();
  const ready = () => { root.classList.add("ready"); sweep(); };
  if (imgAfter.complete && imgAfter.naturalWidth > 0) ready();
  else imgAfter.onload = ready;
}
"""

_COMPARE = st.components.v2.component(
    "pixelopt_compare", html=_COMPARE_HTML, css=_COMPARE_CSS, js=_COMPARE_JS,
    isolate_styles=True,
)


def compare_view(
    *,
    before_url: str,
    after_url: str,
    aspect: float,
    label_before: str,
    label_after: str,
    heat_url: Optional[str] = None,
    worst_box: Optional[Sequence[float]] = None,
    reserve_px: int = 0,
    key: str = "compare",
) -> None:
    """Before/after viewer with slide, flicker and heatmap modes, and zoom.

    `after_url` should carry the encoded bytes verbatim, so what the viewer
    inspects at 4:1 is exactly the file they download.

    `reserve_px` is the vertical space the page needs around the image --
    header, metrics, toolbars. The image box is capped at the window height
    minus that, so the whole result fits on one screen. 0 means no cap.
    """
    stored = st.session_state.get(key)
    saved = getattr(stored, "view", None) if stored is not None else None
    if not isinstance(saved, dict):
        saved = {}
    _COMPARE(
        key=key,
        data={
            "before": before_url,
            "after": after_url,
            "heatmap": heat_url,
            "worst": [float(v) for v in worst_box] if worst_box else None,
            "labelBefore": label_before,
            "labelAfter": label_after,
            "aspect": round(float(aspect), 4),
            "reserve": int(reserve_px),
            "split": saved.get("split", 50),
            "zoom": saved.get("zoom", 1),
            "fitting": saved.get("fitting", True),
            "mode": saved.get("mode", "slide"),
            "seen": saved.get("seen", False),
        },
        on_view_change=lambda: None,
    )
