"""
Streamlit Web App for Deepfake Detection

A beautiful, interactive web interface to detect deepfakes.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging
from model_advanced import AdvancedDeepfakeDetectionModel
from data_preprocessing import ImagePreprocessor

# Setup logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🔍 Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        text-align: center;
        color: #1f77b4;
    }
    .result-real {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 20px 0;
    }
    .result-fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 20px 0;
    }
    .confidence-bar {
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = AdvancedDeepfakeDetectionModel()
        # Use absolute path relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try advanced model first
        advanced_model_path = os.path.join(script_dir, 'models', 'deepfake_advanced_ra.pkl')
        if os.path.exists(advanced_model_path):
            model.load(advanced_model_path)
            return model
        
        # Fallback to basic model
        fallback_path = os.path.join(script_dir, 'models', 'deepfake_random_forest_sklearn.pkl')
        model.load(fallback_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_preprocessor():
    """Load image preprocessor"""
    return ImagePreprocessor(target_size=(256, 256))


def predict_image(image_array, model, preprocessor):
    """Make prediction on image"""
    try:
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3:
            if image_array.max() > 1:
                image_array = image_array / 255.0
        
        # Make prediction using model.predict()
        prediction, probabilities = model.predict(image_array)
        
        if prediction is None:
            return None
        
        return {
            'is_fake': prediction == 1,
            'confidence': max(probabilities) * 100,
            'real_prob': probabilities[0] * 100,
            'fake_prob': probabilities[1] * 100
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


def main():
    # Header
    st.title("🔍 Deepfake Detection System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Upload an image to detect if it's a **deepfake or authentic**.
        
        Our AI model analyzes the image and provides a confidence score.
        """)
    
    with col2:
        st.info("""
        **Supported Formats:**
        - JPG / JPEG
        - PNG
        - BMP
        - TIFF
        """)
    
    st.markdown("---")
    
    # Load model
    model = load_model()
    preprocessor = load_preprocessor()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first:")
        st.code("python train_simple.py --create-sample")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    
    if uploaded_file is not None:
        # Display columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Read and display image
            image = Image.open(uploaded_file)
            st.subheader("📷 Uploaded Image")
            st.image(image, width=400)
        
        with col2:
            st.subheader("⏳ Processing...")
            
            # Convert to numpy array (PIL Image to numpy)
            img_array = np.array(image)
            
            # Ensure RGB format
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            # else: already RGB
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                result = predict_image(img_array, model, preprocessor)
            
            if result:
                st.markdown("---")
                
                # Display result
                if result['is_fake']:
                    st.markdown(f"""
                    <div class="result-fake">
                        <h2>⚠️ FAKE DETECTED</h2>
                        <p>This image appears to be <strong>AI-generated or manipulated</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-real">
                        <h2>✅ REAL / AUTHENTIC</h2>
                        <p>This image appears to be <strong>genuine and authentic</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence display
                st.markdown("### 📊 Confidence Scores")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "Confidence Level",
                        f"{result['confidence']:.2f}%",
                        f"{'Fake' if result['is_fake'] else 'Real'}"
                    )
                
                with col_b:
                    st.metric(
                        "Prediction",
                        "FAKE" if result['is_fake'] else "REAL"
                    )
                
                # Detailed probabilities
                st.markdown("### 🎯 Probability Breakdown")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**🟢 Real:** {result['real_prob']:.2f}%")
                    st.progress(result['real_prob'] / 100)
                
                with col2:
                    st.write(f"**🔴 Fake:** {result['fake_prob']:.2f}%")
                    st.progress(result['fake_prob'] / 100)
                
                st.markdown("---")
                
                # Additional info
                with st.expander("ℹ️ About This Detection"):
                    st.markdown("""
                    **How it works:**
                    - Analyzes image texture and color patterns
                    - Uses Random Forest machine learning
                    - Trained on authentic and fake images
                    - Trained on 40 sample images
                    
                    **For better accuracy:**
                    - Train on larger datasets (1000+ images)
                    - Use diverse image sources
                    - Include different lighting conditions
                    """)
            else:
                st.error("❌ Error making prediction. Please try another image.")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        
        st.subheader("📋 Model Info")
        st.write("""
        - **Type**: Random Forest Classifier
        - **Framework**: scikit-learn
        - **Status**: ✅ Trained and Ready
        """)
        
        st.subheader("🎯 Features")
        st.write("""
        ✅ Real-time detection
        ✅ Confidence scores
        ✅ Supports multiple formats
        ✅ Easy to use
        """)
        
        st.subheader("🔄 Train New Model")
        if st.button("🔁 Retrain Model"):
            st.info("To retrain the model, run:")
            st.code("python train_simple.py --max-images 100")
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit & scikit-learn")


if __name__ == '__main__':
    main()
