import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Dog Disease Predictor", page_icon="🐶", layout="wide")

# Custom CSS to improve the app's appearance
st.markdown("""
<style>
    .reportview-container {
        background: #e0f7fa;  /* Lighter teal background */
    }
    .main {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #00796b;  /* Dark teal color */
    }
    .stButton>button {
        background-color: #00897b;  /* Medium teal color */
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'model_training_notebook/improved_dog_disease_prediction_model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Class indices (update this based on the output from the notebook)
class_indices = {
    0: "Bacterial Dermatosis",
    1: "Fungal Infections",
    2: "Healthy",
    3: "Hypersensitivity Allergic Dermatosis",
    # Add more classes as needed, matching exactly with your training data
}

# Disease descriptions
disease_descriptions = {
    "Bacterial Dermatosis": "A skin infection caused by bacteria, often resulting in redness, swelling, and pustules.",
    "Fungal Infections": "Skin conditions caused by fungi, typically characterized by itching, redness, and scaly patches.",
    "Healthy": "No apparent skin conditions or diseases detected.",
    "Hypersensitivity Allergic Dermatosis": "An allergic reaction affecting the skin, often causing itching, redness, and inflammation.",
}

def preprocess_image(image):
    img = image.resize((299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

st.title("🐶 Dog Disease Prediction App")

st.write("""
This app uses a deep learning model to predict dog diseases from images. 
Upload an image of a dog, and the model will provide a prediction.
""")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
            st.success("Analysis complete!")
            st.write(f"**Prediction:** {class_indices[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            if class_indices[predicted_class] in disease_descriptions:
                st.info(f"**Description:** {disease_descriptions[class_indices[predicted_class]]}")

with col2:
    st.subheader("Disease Information")
    for disease, description in disease_descriptions.items():
        with st.expander(disease):
            st.write(description)

st.markdown("---")
st.subheader("📝 Notes for Accurate Predictions")
st.write("""
For the best results, please ensure that:
1. The uploaded image is clear and well-lit.
2. The dog's affected area (if any) is visible in the image.
3. The image contains only one dog.
4. The image is a recent photograph of the dog's condition.
""")

st.markdown("---")
st.write("### ⚠️ Disclaimer")
st.write("""
This app is for educational purposes only and should not be used as a substitute for professional veterinary advice. 
If you're concerned about your dog's health, please consult with a qualified veterinarian.
""")
