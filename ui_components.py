"""Custom UI for the Streamlit app.

Only one thing here is not native Streamlit, and it earns its place: a
before/after comparison that can zoom. A side-by-side pair of fit-to-screen
images is close to useless for judging compression, because at fit-to-screen a
200 KB and a 500 KB encode of the same photo look identical. The artefacts
live at the pixel level, so the viewer has to be able to get down to 1:1.

Built with Custom Components v2. The trick that keeps it simple: the divider
lives in *viewport* space as a clip on an untransformed wrapper, while the two
image layers share one identical pan/zoom transform. That makes the two views
impossible to get out of alignment -- there is only ever one transform.
"""

from __future__ import annotations

import base64
import io
from typing import Callable, Optional, Tuple

import streamlit as st
from PIL import Image

# The "after" layer is always the real encoded bytes, so artefacts on screen
# are the artefacts in the downloaded file. The "before" layer has to be
# re-encoded for transport; WebP at 92 is visually lossless at this scale.
PREVIEW_MAX_SIDE = 2400
PREVIEW_QUALITY = 92


def _data_url(picture: Image.Image) -> str:
    """Encode the reference layer small enough to ship into the browser."""
    copy = picture.convert("RGB")
    if max(copy.size) > PREVIEW_MAX_SIDE:
        scale = PREVIEW_MAX_SIDE / max(copy.size)
        copy = copy.resize(
            (max(1, int(copy.width * scale)), max(1, int(copy.height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    copy.save(buffer, format="WEBP", quality=PREVIEW_QUALITY, method=4)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def _bytes_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


_HTML = """
<div class="cmp">
  <div class="cmp-view" id="view">
    <div class="cmp-layer" id="layerBefore"><div class="cmp-pan" id="panBefore">
      <img id="imgBefore" alt="reference" draggable="false">
    </div></div>
    <div class="cmp-layer" id="layerAfter"><div class="cmp-pan" id="panAfter">
      <img id="imgAfter" alt="compressed" draggable="false">
    </div></div>
    <div class="cmp-tag cmp-tag-l" id="tagBefore"></div>
    <div class="cmp-tag cmp-tag-r" id="tagAfter"></div>
    <div class="cmp-handle" id="handle"><div class="cmp-grip">
      <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
        <path d="M9.5 7 5 12l4.5 5M14.5 7 19 12l-4.5 5" fill="none"
              stroke="currentColor" stroke-width="2.1"
              stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div></div>
  </div>
  <div class="cmp-bar">
    <div class="cmp-zoom" id="zoomLabel">fit</div>
    <div class="cmp-actions">
      <button type="button" data-zoom="fit">Fit</button>
      <button type="button" data-zoom="1">1:1</button>
      <button type="button" data-zoom="2">2:1</button>
      <button type="button" data-zoom="4">4:1</button>
    </div>
    <div class="cmp-hint">scroll to zoom &middot; drag to pan &middot; drag the divider</div>
  </div>
</div>
"""

_CSS = """
.cmp { width: 100%; font-family: var(--st-font, sans-serif); }

.cmp-view {
  position: relative; width: 100%; aspect-ratio: var(--ar, 1.5);
  overflow: hidden; cursor: grab; touch-action: none;
  background: var(--st-secondary-background-color, #f0f2f6);
  border: 1px solid var(--st-border-color, #d5d9e0);
  border-radius: var(--st-base-radius, 8px);
  /* A checkerboard reads through nothing here, but it makes the frame feel
     like an image surface rather than a blank div while the layers decode. */
  background-image:
    linear-gradient(45deg, rgba(128,128,128,.09) 25%, transparent 25% 75%, rgba(128,128,128,.09) 75%),
    linear-gradient(45deg, rgba(128,128,128,.09) 25%, transparent 25% 75%, rgba(128,128,128,.09) 75%);
  background-size: 22px 22px; background-position: 0 0, 11px 11px;
}
.cmp-view.is-panning { cursor: grabbing; }
.cmp-view.is-sliding { cursor: ew-resize; }

.cmp-layer { position: absolute; inset: 0; }
/* The divider is a clip on this untransformed wrapper, so it stays put in
   viewport space no matter how far the images are panned or zoomed. */
#layerAfter { clip-path: inset(0 0 0 var(--split, 50%)); }

.cmp-pan { position: absolute; inset: 0; transform-origin: 0 0; will-change: transform; }
.cmp-pan img {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain; image-rendering: var(--smoothing, auto);
  user-select: none; -webkit-user-drag: none;
}

.cmp-handle {
  position: absolute; top: 0; bottom: 0; left: var(--split, 50%);
  width: 2px; margin-left: -1px; background: var(--st-primary-color, #ff4b4b);
  box-shadow: 0 0 0 1px rgba(0,0,0,.28); pointer-events: none;
}
.cmp-grip {
  position: absolute; top: 50%; left: 50%; width: 30px; height: 30px;
  transform: translate(-50%, -50%);
  display: grid; place-items: center; border-radius: 50%;
  background: var(--st-primary-color, #ff4b4b); color: #fff;
  box-shadow: 0 2px 9px rgba(0,0,0,.34);
  transition: transform .18s cubic-bezier(.34,1.4,.5,1), box-shadow .18s ease;
}
.cmp-view:hover .cmp-grip { transform: translate(-50%,-50%) scale(1.12); }
.cmp-view.is-sliding .cmp-grip {
  transform: translate(-50%,-50%) scale(.94);
  box-shadow: 0 1px 5px rgba(0,0,0,.4);
}

.cmp-tag {
  position: absolute; top: 10px; padding: 3px 9px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; letter-spacing: .02em;
  color: #fff; background: rgba(15,17,22,.72);
  backdrop-filter: blur(5px); pointer-events: none;
  opacity: 0; transform: translateY(-5px);
  transition: opacity .3s ease .08s, transform .3s ease .08s;
}
.cmp.ready .cmp-tag { opacity: 1; transform: translateY(0); }
.cmp-tag-l { left: 10px; }
.cmp-tag-r { right: 10px; }

.cmp-bar {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  margin-top: 9px; font-size: 12px; color: var(--st-text-color, #31333f);
}
.cmp-zoom {
  font-variant-numeric: tabular-nums; font-weight: 600; min-width: 54px;
  padding: 3px 8px; border-radius: var(--st-base-radius, 8px);
  background: var(--st-secondary-background-color, #f0f2f6);
}
.cmp-actions { display: flex; gap: 5px; }
.cmp-actions button {
  font: inherit; font-size: 11.5px; font-weight: 600; cursor: pointer;
  padding: 4px 11px; color: var(--st-text-color, #31333f);
  background: var(--st-background-color, #fff);
  border: 1px solid var(--st-widget-border-color, var(--st-border-color, #d5d9e0));
  border-radius: var(--st-button-radius, 8px);
  transition: background .16s ease, border-color .16s ease, transform .1s ease;
}
.cmp-actions button:hover {
  border-color: var(--st-primary-color, #ff4b4b);
  background: var(--st-secondary-background-color, #f0f2f6);
}
.cmp-actions button:active { transform: translateY(1px); }
.cmp-actions button[aria-pressed="true"] {
  background: var(--st-primary-color, #ff4b4b); color: #fff;
  border-color: var(--st-primary-color, #ff4b4b);
}
.cmp-hint { margin-left: auto; opacity: .62; }

@media (prefers-reduced-motion: reduce) {
  .cmp-grip, .cmp-tag, .cmp-actions button { transition: none; }
}
"""

_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const q = (sel) => parentElement.querySelector(sel);

  const root = q(".cmp");
  const view = q("#view");
  const panes = [q("#panBefore"), q("#panAfter")];
  const imgBefore = q("#imgBefore");
  const imgAfter = q("#imgAfter");
  const handle = q("#handle");
  const zoomLabel = q("#zoomLabel");
  if (!view || !imgBefore || !imgAfter) return;

  const d = data || {};
  if (imgBefore.src !== d.before) imgBefore.src = d.before || "";
  if (imgAfter.src !== d.after) imgAfter.src = d.after || "";
  q("#tagBefore").textContent = d.labelBefore || "before";
  q("#tagAfter").textContent = d.labelAfter || "after";
  if (d.aspect) view.style.setProperty("--ar", String(d.aspect));

  // Persisted across reruns so changing the budget does not throw away the
  // spot the viewer was inspecting.
  let split = typeof d.split === "number" ? d.split : 50;
  let zoom = typeof d.zoom === "number" ? d.zoom : 1;
  let tx = 0, ty = 0;
  let fitting = d.fitting !== false;

  const setSplit = (v) => {
    split = Math.min(100, Math.max(0, v));
    view.style.setProperty("--split", split + "%");
  };

  const apply = () => {
    // One transform, both layers -- they cannot drift apart.
    const t = `translate(${tx}px, ${ty}px) scale(${zoom})`;
    for (const pane of panes) if (pane) pane.style.transform = t;
    // Nearest-neighbour past 2x so blocking and ringing stay visible instead
    // of being smoothed away by the browser's own interpolation.
    view.style.setProperty("--smoothing", zoom >= 2 ? "pixelated" : "auto");
    zoomLabel.textContent = fitting ? "fit" : zoom.toFixed(zoom < 1 ? 2 : 1) + "x";
    for (const b of parentElement.querySelectorAll(".cmp-actions button")) {
      const z = b.dataset.zoom;
      b.setAttribute("aria-pressed",
        String(z === "fit" ? fitting : !fitting && Math.abs(zoom - Number(z)) < 0.01));
    }
  };

  const clamp = () => {
    // Never let the image pull away from the frame it is being judged in.
    const w = view.clientWidth, h = view.clientHeight;
    const minX = Math.min(0, w - w * zoom), minY = Math.min(0, h - h * zoom);
    tx = Math.min(0, Math.max(minX, tx));
    ty = Math.min(0, Math.max(minY, ty));
  };

  const zoomAbout = (factor, cx, cy) => {
    const next = Math.min(12, Math.max(1, zoom * factor));
    if (Math.abs(next - zoom) < 1e-6) return;
    // Keep the point under the cursor fixed: the whole point of zooming is to
    // look at a specific artefact.
    const k = next / zoom;
    tx = cx - (cx - tx) * k;
    ty = cy - (cy - ty) * k;
    zoom = next;
    fitting = Math.abs(zoom - 1) < 1e-6 && tx === 0 && ty === 0;
    clamp(); apply(); persist();
  };

  let saveTimer = null;
  const persist = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      setStateValue("view", { split, zoom, fitting });
    }, 260);  // debounced: a rerun per wheel tick would be unusable
  };

  view.onwheel = (e) => {
    e.preventDefault();
    const r = view.getBoundingClientRect();
    zoomAbout(e.deltaY < 0 ? 1.16 : 1 / 1.16, e.clientX - r.left, e.clientY - r.top);
  };

  let mode = null, startX = 0, startY = 0, baseTx = 0, baseTy = 0;
  const HANDLE_GRAB_PX = 26;

  view.onpointerdown = (e) => {
    const r = view.getBoundingClientRect();
    const x = e.clientX - r.left;
    const onHandle = Math.abs(x - (split / 100) * r.width) <= HANDLE_GRAB_PX;
    mode = onHandle ? "slide" : "pan";
    view.classList.toggle("is-sliding", mode === "slide");
    view.classList.toggle("is-panning", mode === "pan");
    startX = e.clientX; startY = e.clientY; baseTx = tx; baseTy = ty;
    view.setPointerCapture(e.pointerId);
    if (mode === "slide") { setSplit((x / r.width) * 100); apply(); }
  };

  view.onpointermove = (e) => {
    if (!mode) return;
    const r = view.getBoundingClientRect();
    if (mode === "slide") {
      setSplit(((e.clientX - r.left) / r.width) * 100);
    } else {
      tx = baseTx + (e.clientX - startX);
      ty = baseTy + (e.clientY - startY);
      fitting = false;
      clamp();
    }
    apply();
  };

  const end = (e) => {
    if (!mode) return;
    mode = null;
    view.classList.remove("is-sliding", "is-panning");
    if (e && e.pointerId != null && view.hasPointerCapture(e.pointerId)) {
      view.releasePointerCapture(e.pointerId);
    }
    persist();
  };
  view.onpointerup = end;
  view.onpointercancel = end;

  view.ondblclick = () => {
    zoom = 1; tx = 0; ty = 0; fitting = true; apply(); persist();
  };

  for (const b of parentElement.querySelectorAll(".cmp-actions button")) {
    b.onclick = () => {
      const z = b.dataset.zoom;
      if (z === "fit") { zoom = 1; tx = 0; ty = 0; fitting = true; }
      else { zoom = Number(z); fitting = false; tx = 0; ty = 0; clamp(); }
      apply(); persist();
    };
  }

  setSplit(split);
  apply();
  // Fade the labels in once the layers have actually decoded, so the frame
  // never shows chrome over a blank surface.
  const ready = () => root.classList.add("ready");
  if (imgAfter.complete) ready(); else imgAfter.onload = ready;
}
"""

_COMPARE = st.components.v2.component(
    "pixelopt_compare", html=_HTML, css=_CSS, js=_JS, isolate_styles=True
)


def compare_view(
    before: Image.Image,
    after_bytes: bytes,
    after_mime: str,
    label_before: str,
    label_after: str,
    *,
    key: str = "compare",
    on_view_change: Optional[Callable[[], None]] = None,
):
    """Before/after comparison with a draggable divider and synchronised zoom.

    `after_bytes` is handed over verbatim, so what the viewer inspects at 4:1
    is exactly the file they will download. The reference layer is re-encoded
    for transport and is capped in size -- it is the baseline, not the artefact
    under test.
    """
    stored = st.session_state.get(key)
    saved = getattr(stored, "view", None) if stored is not None else None
    if not isinstance(saved, dict):
        saved = {}

    width, height = before.size
    return _COMPARE(
        key=key,
        data={
            "before": _data_url(before),
            "after": _bytes_url(after_bytes, after_mime),
            "labelBefore": label_before,
            "labelAfter": label_after,
            "aspect": round(max(width, 1) / max(height, 1), 4),
            "split": saved.get("split", 50),
            "zoom": saved.get("zoom", 1),
            "fitting": saved.get("fitting", True),
        },
        on_view_change=on_view_change or (lambda: None),
    )
