import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------
# Load Model
# ----------------------
MODEL_PATH = "rice_leaf_mobilenet.h5"
model = load_model(MODEL_PATH)

# Set your class names here:
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut"
]

# ----------------------
# Preprocess Function
# ----------------------
def preprocess_image(image):
    image = image.resize((224, 224))   # Change size to match your model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------
# Prediction Function
# ----------------------
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    return CLASS_NAMES[class_index], confidence

# ----------------------
# Streamlit UI
# ----------------------
st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload a rice leaf image to predict the disease.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    label, confidence = predict(image)

    st.success(f"### Predicted Disease: **{label}**")
    st.info(f"**Confidence:** {confidence*100:.2f}%")
