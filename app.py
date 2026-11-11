
import streamlit as st
import onnxruntime as ort
import numpy as np

from PIL import Image, ImageOps

# Class names in training order
CLASS_NAMES = [
    "Early Blight",
    "Healthy",
    "Late Blight"
]

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "tomato_disease_model.onnx",
        providers=["CPUExecutionProvider"]
    )

def preprocess_image(image):
    image = image.convert("RGB")

    # Resize and center crop to 224x224
    image = ImageOps.fit(
        image,
        (224, 224),
        method=Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # Normalize
    image_array = (image_array - MEAN) / STD

    # HWC -> CHW and add batch dimension
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)

def predict(image, session):
    input_name = session.get_inputs()[0].name
    input_data = preprocess_image(image)

    logits = session.run(
        None,
        {input_name: input_data}
    )[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    predicted_index = np.argmax(probabilities)

    return CLASS_NAMES[predicted_index], probabilities

# Streamlit interface
st.title("🍅 Tomato Leaf Disease Classifier")
st.write("Upload a tomato leaf image to detect its condition.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Predict"):
        session = load_model()
        prediction, probabilities = predict(image, session)

        st.success(f"Prediction: {prediction}")

        st.subheader("Class Probabilities")

        for class_name, probability in zip(CLASS_NAMES, probabilities):
            st.write(f"{class_name}: {probability:.2%}")
            st.progress(float(probability))
