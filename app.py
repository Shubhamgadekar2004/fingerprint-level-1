import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION (Must match training script)
# ==========================================
# ⚠️ UPDATE THIS PATH IF YOUR MODEL FILE HAS MOVED
MODEL_PATH = "fingerprint_best_model.pth"

NUM_CLASSES = 3         
# Order must match your LABEL_DICT (0, 1, 2) from the training script
CLASS_NAMES = ['Arch', 'Whorl', 'Loop'] 
IMG_SIZE = 224

# Normalization constants used in val_transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 MODEL DEFINITION (Crucial: Must match training script exactly)
# ==========================================
class FingerprintClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name='efficientnet_b0'):
        super(FingerprintClassifier, self).__init__()
        
        # This architecture definition MUST be present for torch.load to work
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=None) 
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            # Fallback is unnecessary but good practice
            raise ValueError(f"Model name '{model_name}' not supported in dashboard class.")

    def forward(self, x):
        return self.model(x)

# ==========================================
# 💾 MODEL LOADING (FIXED for custom class serialization)
# ==========================================
@st.cache_resource
def load_trained_model():
    """
    Loads the full model object saved via torch.save(model).
    The weights_only=False flag bypasses the security check for custom classes.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            return None

        # 1. Load the entire model object, setting weights_only=False 
        # to allow loading of the custom FingerprintClassifier class definition.
        model = torch.load(
            MODEL_PATH, 
            map_location=DEVICE,
            weights_only=False # <--- FIXED: Allows loading of the custom class object
        )
        
        # 2. Ensure it is set to evaluation mode and on the correct device
        model.to(DEVICE)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Critical Error loading model: {e}")
        st.warning("Hint: Check the model path and ensure the FingerprintClassifier class is defined above.")
        return None

# ==========================================
# 🖼️ IMAGE PREPROCESSING
# ==========================================
def preprocess_image(image):
    """
    Applies the validation transforms from the training script.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Crucial: Convert to RGB, matching your training script: .convert('RGB')
    image = image.convert('RGB')
    
    # Apply transforms and add batch dimension (1, C, H, W)
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

# ==========================================
# 📱 MAIN DASHBOARD UI
# ==========================================
st.set_page_config(page_title="EfficientNet Fingerprint Classifier", page_icon="🔍", layout="wide")

st.title("🔍 Fingerprint Pattern Classification (EfficientNet-B0)")
st.write(f"Model loaded from: `{MODEL_PATH}` | Running on **{DEVICE}**")

# Load Model
model = load_trained_model()

if model:
    st.success("✅ Model (EfficientNet-B0) loaded successfully!")
    
    uploaded_file = st.file_uploader("Upload a Fingerprint Image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display Image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Fingerprint', use_column_width=True)

        with col2:
            st.subheader("Classification Prediction")
            
            # Run Inference
            input_tensor = preprocess_image(image).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            # Get Results
            predicted_label = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # Display Prediction
            st.metric(label="Predicted Pattern", value=predicted_label)
            st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
            
            # Visual Indicator
            if confidence_score > 90:
                st.balloons()
                st.success("⭐ High Confidence Match!")
            elif confidence_score > 70:
                st.warning("🔔 Moderate Confidence - Reliable")
            else:
                st.error("⚠️ Low Confidence - Review Recommended")

            st.divider()
            st.subheader("Probability Distribution")
            
            # Create Dataframe for chart
            probs_np = probabilities.cpu().numpy()
            chart_data = pd.DataFrame({
                "Pattern": CLASS_NAMES,
                "Probability": probs_np
            })
            
            # Display chart
            st.bar_chart(chart_data, x="Pattern", y="Probability", color="#0072B2")

else:
    st.info("The model could not be loaded. Please ensure the path is correct and the FingerprintClassifier class is defined above.")