"""Objects: name what is in an image, remove one thing, predict what was behind it."""

from __future__ import annotations

import re

import streamlit as st
from PIL import Image

from app_shared import (
    UPLOAD_TYPES,
    array_url,
    fingerprint,
    png_bytes,
    vision_classical,
    vision_find,
    vision_generated,
    vision_image,
    vision_mask,
    vision_scene,
    vision_suggestion,
)
from pixelopt.vision import detect, draw, models, reason, segment
from ui_components import compare_view, file_line, intro_cards, page_header, workbench

# Vertical space this page needs around the result viewer: the file line, the
# caption, the view switch, the viewer's toolbar, the note and the buttons.
# Measured at 1440 x 900.
VIEWER_RESERVE_PX = 400
# The labelled picture gets about the same height as the viewer.
PICTURE_MAX_HEIGHT = 520
# What the fill model is told when nobody says what is behind the object.
DEFAULT_FILL = "background, a seamless continuation of the surroundings"
FILL_CHOICES = {models.SDXL: "SDXL · sharper", models.SD15: "SD 1.5 · faster"}
SETUP = ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n"
         "pip install -r requirements-vision.txt")


def upload_widget():
    # Identical arguments in both layouts, so the file survives the move.
    return st.file_uploader("Image", type=UPLOAD_TYPES, key="objects_upload")


status = models.availability()

if st.session_state.get("objects_upload") is None:
    page_header(
        eyebrow="Recognise, remove, predict",
        title="Name what is in a photo. Take one thing out.",
        subtitle=(
            "PixelOpt lists what it sees, outlines the thing you pick, and fills the gap "
            "two ways: from the surroundings, and with what a model predicts was behind it. "
            "Everything runs on this computer."
        ),
    )
    upload_widget()
    intro_cards([
        ("name", ":material/label:", "Named, then outlined",
         "A vision-language model lists what is in the photo. A segmentation model then "
         "traces the exact outline of the one you choose."),
        ("guess", ":material/psychology:", "A prediction, not a recovery",
         "The camera never saw what was behind the object. The generated fill is one "
         "plausible picture of it, and a different one each time you ask."),
        ("baseline", ":material/compare:", "Measured against a baseline",
         "Classical inpainting fills the same hole from its edges, so you can see what "
         "the learned model adds and what it invents."),
    ])
    if not status.installed:
        st.info("This page needs the optional vision libraries, which are not installed yet. "
                "Upload an image for the setup steps.", icon=":material/extension:")
    st.stop()

uploaded = st.session_state["objects_upload"]
raw = uploaded.getvalue()
key = fingerprint(raw)
image, source_size = vision_image(raw)
height, width = image.shape[:2]
downsized = source_size != (width, height)

file_line(uploaded.name,
          f"{width} × {height}"
          + (f" working copy of {source_size[0]} × {source_size[1]}" if downsized else ""))

if not status.installed:
    upload_widget()
    st.warning("The Objects page needs the optional vision libraries. Missing: "
               + ", ".join(status.missing) + ". Install them into this environment, then reload.",
               icon=":material/extension:")
    st.code(SETUP, language="bash")
    st.stop()

panel, result_col = workbench("objects")

# --------------------------------------------------------- what is in it

with panel:
    upload_widget()
    try:
        with st.spinner("Naming what is in the image. The first run downloads the model…"):
            scene = vision_scene(raw)
    except OSError as error:
        # Almost always the first-run download: no connection, or no disk space.
        st.error(f"The naming model could not be loaded: {error}", icon=":material/error:")
        st.stop()

    extras = st.session_state.setdefault("objects_extra", {})
    with st.form("objects_find", border=False):
        query = st.text_input("Find something else", placeholder="leaves",
                              help="Looks for a name that is not in the list, such as a part "
                                   "of something or a kind of surface.")
        search = st.form_submit_button("Find", icon=":material/search:")
    if search and detect.clean_name(query):
        with st.spinner("Looking for it…"):
            hits = vision_find(raw, detect.clean_name(query))
        listed = detect.merge([scene.objects, extras.get(key, [])])
        fresh = [hit for hit in hits if detect.already_listed(hit, listed) is None]
        if fresh:
            extras[key] = list(extras.get(key, [])) + fresh
        elif hits:
            # The detector would rather box something than nothing, so a hit on
            # a place that is already listed is reported, not added.
            same = detect.already_listed(hits[0], listed)
            st.caption(f":material/info: That points at **{same.name}**, which is already "
                       "in the list.")
        else:
            st.caption(":material/search_off: Nothing by that name was found in this image.")

    objects = detect.number(detect.merge([scene.objects, extras.get(key, [])]))
    names = [item.name for item in objects]
    if names:
        pick = st.pills("Found in the image", names, selection_mode="single",
                        key=f"objects_pick_{key}")
    else:
        pick = None
        st.caption(":material/visibility_off: Nothing was recognised. Try the box above.")

chosen = objects[names.index(pick)] if pick in names else None

# ------------------------------------------------------ the one to remove

mask = None
grow_px = 0
behind = ""
fill_model = models.FILL_MODELS[0]
remove = False
if chosen is not None:
    with panel:
        grow_px = st.slider(
            "Grow outline", 0, 40, 12, format="%d px", key="objects_grow",
            help="Widens the outline so no rim or shadow of the object is left behind. "
                 "The fill is blended back into the photo inside this margin.",
        )
        with st.spinner("Outlining it…"):
            mask = segment.grow(vision_mask(raw, chosen.box), grow_px)

        # A cold answer from the local model takes 10-30 s: it is loaded for the
        # question and unloaded straight after, to give the GPU back. So it is
        # asked for, not run on every selection. The request is carried across
        # a rerun because the text box's value can only be set before the box
        # is drawn.
        reader = reason.pick_vision_model(reason.models())
        behind_key = f"objects_behind_{key}_{chosen.box}"
        if st.session_state.pop("objects_suggest", None) == behind_key and reader:
            with st.spinner(f"Asking {reader} what is behind it…"):
                suggestion = vision_suggestion(raw, chosen.name, chosen.box, reader)
            if suggestion:
                st.session_state[behind_key] = suggestion
            else:
                st.caption(":material/help: The local model gave no usable answer. Type your own.")
        behind = st.text_input(
            "What is behind it?", placeholder="bare branches against the sky", key=behind_key,
            help="The generated fill paints this. Leave it empty to continue the surroundings.",
        )
        if reader:
            if st.button("Suggest", icon=":material/psychology:", key="objects_ask",
                         help=f"Asks {reader}, running on this computer, what the object is "
                              "attached to or hiding. Takes up to half a minute."):
                st.session_state["objects_suggest"] = behind_key
                st.rerun()
        else:
            st.caption(":material/info: Start Ollama with a vision model to have this "
                       "suggested, or type your own.")
        if status.cuda:
            fill_model = st.segmented_control(
                "Fill model", list(models.FILL_MODELS), default=models.FILL_MODELS[0],
                format_func=FILL_CHOICES.get, key="objects_model",
                help="SDXL paints at 1024 px, takes about 20 seconds a fill on an 8 GB card, "
                     "and follows the description closely. SD 1.5 paints at 512 px in about "
                     "6 seconds with half the memory, but needs it spelled out: it paints "
                     "leaves back onto 'branches of a tree' and wants 'bare branches'.",
            ) or models.FILL_MODELS[0]
        remove = st.button("Remove", type="primary", icon=":material/auto_fix_high:",
                           key="objects_remove", width="stretch")
        if not status.cuda:
            st.caption(":material/memory: No CUDA GPU was found, so only the classical fill "
                       "is available.")

jobs = st.session_state.setdefault("objects_job", {})
if chosen is not None and remove:
    jobs[key] = {"name": chosen.name, "box": chosen.box, "grow": grow_px,
                 "prompt": behind.strip() or DEFAULT_FILL, "seed": 0, "model": fill_model}
    st.session_state["objects_view"] = "Generated" if status.cuda else "Classical"
job = jobs.get(key)

# ------------------------------------------------------------------ result

with result_col:
    if scene.caption:
        st.caption(scene.caption)

    view = "Objects"
    if job is not None:
        views = ["Objects"] + (["Generated"] if status.cuda else []) + ["Classical"]
        # The Remove button sets this key to jump to the result, so the widget
        # takes its value from session state and has no default of its own.
        if st.session_state.get("objects_view") not in views:
            st.session_state["objects_view"] = "Objects"
        view = st.segmented_control("View", views, key="objects_view",
                                    label_visibility="collapsed") or "Objects"

    if view == "Objects":
        picture = draw.overlay(image, objects, names.index(pick) if chosen is not None else None, mask)
        st.image(picture, width=min(width, int(PICTURE_MAX_HEIGHT * width / height)))
        if chosen is None:
            st.caption(":material/touch_app: Pick something in the list to outline it.")
        else:
            st.caption(f"**{chosen.name}** covers {segment.share(mask) * 100:.1f}% of the image, "
                       f"with its outline grown by {grow_px} px.")
    else:
        result = None
        if view == "Classical":
            result = vision_classical(raw, job["box"], job["grow"])
            note = ("Classical fill (OpenCV, Telea's method): the surrounding colours are "
                    "carried inwards. It cannot invent structure.")
        else:
            try:
                with st.spinner("Painting what could be there. The first run loads the model, "
                                "which takes a while…"):
                    made = vision_generated(raw, job["box"], job["grow"], job["prompt"],
                                            job["seed"], job["model"])
                result = made["image"]
                note = (f"Generated by {made['model']} · seed {job['seed']} · "
                        f"{made['seconds']:.0f} s · “{job['prompt']}”. **A plausible guess, "
                        "not a recovery:** ask again and it changes.")
            except (RuntimeError, OSError) as error:
                st.error(f"The generated fill did not run: {error}", icon=":material/error:")

        if result is not None:
            compare_view(
                before_url=array_url(image),
                after_url=array_url(result),
                aspect=width / height,
                label_before="Original",
                label_after=f"{view} · without {job['name']}",
                reserve_px=VIEWER_RESERVE_PX,
                key="objects_compare",
            )
            st.caption(note)
            current = {"name": chosen.name, "box": chosen.box, "grow": grow_px} if chosen else {}
            if any(job[field] != current.get(field) for field in ("name", "box", "grow")):
                st.caption(":material/history: The selection or outline has changed since this "
                           "result. Press Remove to update it.")
            stem = uploaded.name.rsplit(".", 1)[0]
            slug = re.sub(r"[^a-z0-9]+", "_", job["name"].lower()).strip("_") or "object"
            with st.container(horizontal=True, gap="small"):
                # A callable runs only when clicked, so nothing is encoded per rerun.
                st.download_button(
                    "Download PNG", data=lambda: png_bytes(Image.fromarray(result)),
                    file_name=f"{stem}_without_{slug}.png", mime="image/png",
                    type="primary", icon=":material/download:", on_click="ignore",
                )
                if view == "Generated" and st.button("Try another guess", icon=":material/casino:",
                                                     key="objects_reroll"):
                    job["seed"] += 1
                    st.rerun()
