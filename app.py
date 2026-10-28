"""Waste Classification CNN dashboard for Streamlit Community Cloud.

Place app.py, requirements.txt, best_waste_cnn.keras and class_names.json
in the same GitHub repository. The model is loaded for inference only.
"""

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage

st.set_page_config(page_title="AI Waste Classification", page_icon="♻️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_waste_cnn.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
LABELS = {"O": "Organic Waste", "R": "Recyclable Waste"}

st.markdown("""<style>
.block-container {max-width: 1100px; padding-top: 2rem}
.hero {padding: 1.6rem; background: linear-gradient(115deg,#12634b,#219178); color: white;
       border-radius: 18px; margin-bottom: 1.5rem}
.hero h1 {color: white; margin: 0 0 .3rem 0}
.hero p {margin: 0; color: #e8fff4}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>♻️ AI Waste Classification Dashboard</h1>'
            '<p>Upload an image, explore CNN predictions, and download a PDF report.</p></div>',
            unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_labels():
    with CLASS_PATH.open(encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("class_names.json must be a non-empty JSON list of strings.")
    return names

def display_label(raw_label):
    return LABELS.get(raw_label, raw_label.replace("_", " ").title())

def prepare_image(image, width, height):
    """Match Colab training: RGB, resize, float32 pixels in 0..255.

    Do NOT divide by 255 here: the trained CNN contains Rescaling(1/255).
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)

def create_pdf(image, filename, predicted, confidence, scores, timestamp):
    """Create an in-memory, downloadable one-page prediction report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [Paragraph("AI Waste Classification - Prediction Report", styles["Title"]),
             Spacer(1, 14),
             Paragraph(f"Generated: {timestamp}", styles["Normal"]),
             Paragraph(f"Image filename: {filename}", styles["Normal"]),
             Spacer(1, 16)]
    picture = ImageOps.exif_transpose(image).convert("RGB")
    picture.thumbnail((800, 800))
    image_bytes = io.BytesIO()
    picture.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    iw, ih = picture.size
    max_w, max_h = 320, 230
    factor = min(max_w / iw, max_h / ih)
    story.extend([PDFImage(image_bytes, width=iw * factor, height=ih * factor),
                  Spacer(1, 18),
                  Paragraph(f"Predicted category: <b>{predicted}</b>", styles["Heading2"]),
                  Paragraph(f"Model confidence: <b>{confidence:.2%}</b>", styles["Normal"]),
                  Spacer(1, 12)])
    rows = [["Waste category", "Model probability"]]
    rows.extend([[name, f"{score:.2%}"] for name, score in scores])
    table = Table(rows, colWidths=[290, 170])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176e59")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d8e2df")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eff7f4")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([table, Spacer(1, 20),
                  Paragraph("Note: Confidence is a model probability, not a guarantee of correctness. "
                            "Predictions are for educational demonstration and should be reviewed by a person.",
                            styles["Normal"]),
                  Spacer(1, 16),
                  Paragraph("Universiti Malaysia Perlis (UniMAP) | AI Practical Class", styles["Normal"])])
    doc.build(story)
    return buffer.getvalue()

try:
    if not MODEL_PATH.is_file() or not CLASS_PATH.is_file():
        missing = [p.name for p in (MODEL_PATH, CLASS_PATH) if not p.is_file()]
        st.error("Missing deployment file(s): " + ", ".join(missing))
        st.info("Put the trained model and class_names.json beside app.py in your GitHub repository.")
        st.stop()
    model = load_model()
    class_names = load_labels()
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    image_height, image_width = int(shape[1]), int(shape[2])
    if shape[-1] != 3:
        raise ValueError("This dashboard expects a three-channel RGB image model.")
    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output count does not match class_names.json.")
except Exception as exc:
    st.error("Unable to load the CNN model or class labels.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.write("**Method:** Convolutional Neural Network (CNN)")
    st.write(f"**Input image:** {image_width} x {image_height} RGB")
    st.write(f"**Categories:** {len(class_names)}")
    for i, name in enumerate(class_names):
        st.write(f"{i}: {display_label(name)}")
    st.divider()
    st.caption("This model was trained for an educational practical using organic and recyclable waste images.")
    st.caption("Ir. Dr. Marni Azira Markom | Universiti Malaysia Perlis")

st.subheader("1. Upload a waste image")
uploaded = st.file_uploader("Choose an image (JPG, PNG or WebP)",
                            type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Upload one image to view its classification and generate a report.")
    st.stop()

try:
    uploaded.seek(0)
    image = Image.open(uploaded)
    image.load()
except (UnidentifiedImageError, OSError, ValueError):
    st.error("The uploaded file is not a valid supported image. Please try another file.")
    st.stop()

left, right = st.columns(2, gap="large")
with left:
    st.subheader("2. Uploaded sample")
    st.image(image, caption=uploaded.name, use_container_width=True)

try:
    tensor = prepare_image(image, image_width, image_height)
    raw_scores = np.asarray(model.predict(tensor, verbose=0)[0], dtype=np.float64)
    if raw_scores.ndim != 1 or len(raw_scores) != len(class_names):
        raise ValueError("Unexpected CNN prediction shape.")
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("The CNN returned invalid prediction values.")
    # The notebook's final layer uses softmax: outputs are probabilities.
    if np.any(raw_scores < -1e-5) or np.any(raw_scores > 1.00001) or not np.isclose(raw_scores.sum(), 1, atol=1e-3):
        raise ValueError("Model outputs are not softmax probabilities. Check the trained model.")
    predicted_index = int(np.argmax(raw_scores))
    confidence = float(raw_scores[predicted_index])
    predicted_name = display_label(class_names[predicted_index])
except Exception as exc:
    st.error("Prediction failed. Please verify the model and uploaded image.")
    st.exception(exc)
    st.stop()

with right:
    st.subheader("3. CNN prediction")
    st.metric("Predicted waste category", predicted_name)
    st.metric("Prediction confidence", f"{confidence:.2%}")
    st.progress(float(confidence))
    st.caption("Confidence is the CNN's softmax probability, not a guarantee of correctness.")

st.subheader("4. Probability by category")
scores = [(display_label(name), float(raw_scores[i])) for i, name in enumerate(class_names)]
for name, probability in sorted(scores, key=lambda x: x[1], reverse=True):
    st.write(f"**{name}: {probability:.2%}**")
    st.progress(float(probability))

st.divider()
st.subheader("5. Download prediction report")
timestamp = datetime.now().astimezone().strftime("%d %B %Y, %I:%M %p %Z")
try:
    pdf_bytes = create_pdf(image, uploaded.name, predicted_name, confidence, scores, timestamp)
    st.download_button("📄 Download PDF Report", data=pdf_bytes,
                       file_name="waste_classification_report.pdf", mime="application/pdf",
                       type="primary")
except Exception as exc:
    st.error("Unable to generate the PDF report.")
    st.exception(exc)

st.caption("AI Waste Classification Practical | Universiti Malaysia Perlis (UniMAP)")
