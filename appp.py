import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load fine-tuned model
model = tf.keras.models.load_model("rice_leaf_model.h5")

# Define class names (update if needed)
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Streamlit UI
st.title("🌾 Rice Leaf Disease Classifier")
st.write("Upload a rice leaf image and get disease prediction instantly.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Show all class probabilities
        st.subheader("Class Probabilities:")
        for i, score in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {score * 100:.2f}%")
