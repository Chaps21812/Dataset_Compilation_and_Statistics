import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Multi-Image Bounding Boxes", layout="wide", page_icon="💫")
st.title("💫 Star Streak Annotation")

# --- Initialize session_state ---
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []  # list of {"name": ..., "image": ...}
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "annotations_dict" not in st.session_state:
    st.session_state.annotations_dict = {}

# --- Upload images ---
uploaded_files = st.file_uploader(
    "Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        # Only add if this file name is not already uploaded
        if file.name not in [img["name"] for img in st.session_state.uploaded_images]:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.uploaded_images.append({"name": file.name, "image": image})

# --- Stop if no images ---
if not st.session_state.uploaded_images:
    st.warning("Please upload at least one image.")
    st.stop()

# --- Navigation buttons ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅ Previous"):
        st.session_state.current_index = max(0, st.session_state.current_index - 1)
with col3:
    if st.button("Next ➡"):
        st.session_state.current_index = min(len(st.session_state.uploaded_images) - 1,
                                            st.session_state.current_index + 1)

current_img_data = st.session_state.uploaded_images[st.session_state.current_index]
st.write(f"Viewing image {st.session_state.current_index + 1} of {len(st.session_state.uploaded_images)}: **{current_img_data['name']}**")
image = current_img_data["image"]
image_pil = Image.fromarray(image)

# --- Drawable canvas ---
existing_objects = st.session_state.annotations_dict.get(st.session_state.current_index, [])

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=image_pil,
    update_streamlit=True,
    height=image.shape[0],
    width=image.shape[1],
    drawing_mode="rect",
    key=f"canvas_{st.session_state.current_index}",
    initial_drawing=existing_objects
)

# --- Save annotations & display centers ---
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    st.session_state.annotations_dict[st.session_state.current_index] = objects

    annotated = image.copy()
    centers = []

    for obj in objects:
        left = obj["left"]
        top = obj["top"]
        width = obj["width"]
        height = obj["height"]

        cx = left + width / 2
        cy = top + height / 2
        centers.append((cx, cy))
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 255, 0), -1)

    st.image(annotated, caption="Centers of all drawn bounding boxes")
    st.write("Coordinates of centers:")
    for i, (cx, cy) in enumerate(centers, 1):
        st.write(f"{i}: ({cx:.1f}, {cy:.1f})")
else:
    st.info("No drawing yet for this image.")
