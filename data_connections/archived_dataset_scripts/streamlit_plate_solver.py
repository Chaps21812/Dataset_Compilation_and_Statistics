import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from src.raw_datset import raw_dataset
from astropy.io import fits
import os
from src.preprocess_functions import iqr_log
import numpy as np
from src.astrometric_localization import match_to_catalogue

def run_plate_solve():
    if len(st.session_state.annotations[img_path]) <0:
        return None
    else:
        idx = st.session_state.current_idx
        annot_path = annotation_paths[idx]
        img_path = st.session_state.dataset.annotation_to_fits[annot_path]
        match_to_catalogue(st.session_state.annotations[img_path],scales={3,4,5})

directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CalSatLMNT01-2024"
dataset = raw_dataset(directory)

st.set_page_config(layout="wide", page_icon="💫")

# Initialize session state
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = {}
if "star_annotations" not in st.session_state:
    st.session_state.star_annotations = {}
if "initialized" not in st.session_state:
    st.session_state.initialized = {}
if "dataset" not in st.session_state:
    st.session_state.dataset = raw_dataset(directory)
if "canvas_initialized" not in st.session_state:
    st.session_state.canvas_initialized = {}

annotation_paths = list(st.session_state.dataset.annotation_to_fits.keys())

# Get current image
idx = st.session_state.current_idx
annot_path = annotation_paths[idx]
img_path = st.session_state.dataset.annotation_to_fits[annot_path]
hdu = fits.open(img_path)
hdul = hdu[0]
img = iqr_log(hdul.data)
arr = np.transpose(img, (1, 2, 0))  # now shape is H,W,3
# Create a Pillow image in grayscale
img = Image.fromarray(arr)


st.header(f"Annotate Image {idx+1} / {len(annotation_paths)}: {os.path.basename(img_path)}")

colA, colB = st.columns([6,1])

with colA:
    # Buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Prev Image"):
            if idx > 0:
                st.session_state.current_idx -= 1
                st.experimental_rerun()
    with col2:
        if st.button("Next Image"):
            if idx < len(annotation_paths) - 1:
                st.session_state.current_idx += 1
                st.experimental_rerun()
                
    with col3:
        if st.button("Run Plate solve"):
            st.success("Processing all annotated dots...")
            for img_path, dots in st.session_state.annotations.items():
                st.write(f"Image: {img_path}")
                for dot in dots:
                    st.write(f"Dot coordinates: {dot}")
    with col4:
        if st.button("Save WCS"):
            st.success("Processing all annotated dots...")
            for img_path, dots in st.session_state.annotations.items():
                st.write(f"Image: {img_path}")
                for dot in dots:
                    st.write(f"Dot coordinates: {dot}")
    with col5:
        if st.button("Clear annotations"):
            st.success("Processing all annotated dots...")
            for img_path, dots in st.session_state.annotations.items():
                st.write(f"Image: {img_path}")
                for dot in dots:
                    st.write(f"Dot coordinates: {dot}")

    # # Only set initial_drawing if this image hasn’t been initialized yet
    if not st.session_state.canvas_initialized.get(img_path, False):
        initial = st.session_state.canvas_data.get(img_path, None)
        st.session_state.canvas_initialized[img_path] = True
    else:
        initial = None  # prevent reloading on every rerun

    # initial = st.session_state.canvas_data.get(img_path, None)

    # Only load initial drawing once per image

    img_width, img_height = img.size
    max_width = 1200  # maximum width of the canvas
    scale = min(1.0, max_width / img_width)
    canvas_result = st_canvas(
        background_image=img,
        height=int(img_height * scale),
        width=int(img_width * scale),
        drawing_mode="point",
        initial_drawing=initial,
        update_streamlit=False,
        stroke_width=5,
        stroke_color="red",
        key=f"canvas_{img_path}"
    )

    if st.button("bruh"):
        print("bruh")

    # Store dots for this image
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        dots = [(int(obj["left"]), int(obj["top"])) for obj in objects if obj["type"] == "circle"]
        st.session_state.canvas_data[img_path] = canvas_result.json_data
        st.session_state.annotations[img_path] = dots


with colB:
    st.write("Stars annotated:", st.session_state.annotations.get(img_path, []))
