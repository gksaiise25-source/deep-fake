"""
Streamlit Web Application for Deepfake Detection

Interactive web interface for uploading and detecting deepfakes
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image

from predict import DeepfakePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 20px;
        }
        .stButton > button {
            width: 100%;
            padding: 10px;
            font-size: 16px;
        }
        .result-real {
            background-color: #d4edda;
            color: #155724;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }
        .result-fake {
            background-color: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)


def load_model(model_path: str) -> DeepfakePredictor:
    """Load the deepfake detector model"""
    try:
        predictor = DeepfakePredictor(
            model_path=model_path,
            model_type='efficientnet',
            input_shape=(256, 256)
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def display_prediction_result(result: dict, col):
    """Display prediction result"""
    with col:
        if result['is_fake']:
            st.markdown(f"""
                <div class="result-fake">
                    <h3>⚠️ FAKE DETECTED</h3>
                    <p>This image appears to be AI-generated or manipulated.</p>
                    <p><strong>Confidence: {result['confidence']:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-real">
                    <h3>✅ REAL</h3>
                    <p>This image appears to be authentic.</p>
                    <p><strong>Confidence: {result['confidence']:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    
    # Title and description
    st.title("🔍 Deepfake Detection System")
    st.markdown("""
        This application uses Artificial Intelligence to detect whether images or videos 
        are authentic or AI-generated (deepfakes). Upload a file and get instant results 
        with confidence scores.
    """)
    
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model:",
            ["EfficientNet (Recommended)", "ResNet50", "Xception", "Custom CNN"]
        )
        
        st.info("""
            **Model Information:**
            - **EfficientNet**: Faster, good accuracy
            - **ResNet50**: Balanced performance
            - **Xception**: Slower but highest accuracy
            - **Custom CNN**: Lightweight, for testing
        """)
        
        st.divider()
        
        # Detection options
        st.subheader("Detection Options")
        detect_face = st.checkbox("Enable Face Detection", value=True, 
                                  help="Automatically detect and extract faces")
        visualize = st.checkbox("Show Visualization", value=True)
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About This App")
        st.info("""
            **Supported Formats:**
            - Images: JPG, PNG, BMP, TIFF
            - Videos: MP4, AVI, MOV, MKV
            
            **How it works:**
            1. Upload an image or video
            2. AI analyzes the content
            3. Get prediction: REAL or FAKE
            4. Confidence score shows certainty
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Content")
        
        # Tabs for different input types
        tab1, tab2 = st.tabs(["Image", "Video"])
        
        with tab1:
            uploaded_image = st.file_uploader(
                "Choose an image",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image to detect if it's a deepfake"
            )
            
            if uploaded_image is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(uploaded_image.getbuffer())
                    temp_path = tmp.name
                
                try:
                    # Display uploaded image
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Analyze button
                    if st.button("🔍 Analyze Image", key="analyze_image"):
                        with st.spinner("Analyzing image..."):
                            # Load model
                            model_path = get_model_path(model_type)
                            predictor = load_model(model_path)
                            
                            if predictor:
                                # Make prediction
                                result = predictor.predict_image(
                                    temp_path,
                                    detect_face=detect_face
                                )
                                
                                if result:
                                    st.session_state.image_result = result
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        with tab2:
            uploaded_video = st.file_uploader(
                "Choose a video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video to detect if it contains deepfakes"
            )
            
            if uploaded_video is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_video.getbuffer())
                    temp_path = tmp.name
                
                try:
                    # Display video info
                    st.video(uploaded_video)
                    
                    # Analyze button
                    if st.button("🔍 Analyze Video", key="analyze_video"):
                        with st.spinner("Analyzing video (this may take a moment)..."):
                            # Load model
                            model_path = get_model_path(model_type)
                            predictor = load_model(model_path)
                            
                            if predictor:
                                # Make prediction
                                result = predictor.predict_video(
                                    temp_path,
                                    max_frames=30,
                                    detect_face=detect_face
                                )
                                
                                if result:
                                    st.session_state.video_result = result
                
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Results display
    with col2:
        st.subheader("📊 Results")
        
        if 'image_result' in st.session_state:
            result = st.session_state.image_result
            display_prediction_result(result, st)
            
            if visualize:
                st.subheader("Visualization")
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                axes[0].imshow(result['original_image'])
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(result['processed_image'])
                title = f"{result['label']}\nConfidence: {result['confidence']:.2f}%"
                color = 'red' if result['is_fake'] else 'green'
                axes[1].set_title(title, color=color, fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                st.pyplot(fig)
            
            # Details
            with st.expander("📋 Details"):
                st.write(f"**Face Detected:** {result['face_detected']}")
                st.write(f"**Raw Prediction Score:** {result['raw_prediction']:.4f}")
                st.write(f"**Confidence:** {result['confidence']:.2f}%")
        
        elif 'video_result' in st.session_state:
            result = st.session_state.video_result
            display_prediction_result(result, st)
            
            if visualize:
                st.subheader("Video Analysis")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Frames Analyzed", result['total_frames_processed'])
                with col_b:
                    st.metric("Fake Frames", f"{result['fake_frame_percentage']:.1f}%")
                with col_c:
                    st.metric("Overall Confidence", f"{result['confidence']:.2f}%")
                
                # Frame-by-frame plot
                predictions = result['frame_predictions']
                frames = [p['frame'] for p in predictions]
                confidences = [p['confidence'] for p in predictions]
                is_fakes = [p['is_fake'] for p in predictions]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['red' if f else 'green' for f in is_fakes]
                ax.bar(frames, confidences, color=colors, alpha=0.7)
                ax.axhline(y=50, color='black', linestyle='--', label='Decision Threshold')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Confidence (%)')
                ax.set_title('Per-Frame Prediction Confidence')
                ax.legend()
                ax.set_ylim([0, 100])
                st.pyplot(fig)
            
            # Details
            with st.expander("📋 Details"):
                st.write(f"**Total Frames:** {result['total_frames_processed']}")
                st.write(f"**Fake Frames:** {result['fake_frame_percentage']:.1f}%")
        
        else:
            st.info("👈 Upload and analyze content to see results")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
            <p style='text-align: center; color: gray;'>
            Made with ❤️ using Streamlit and TensorFlow
            </p>
        """, unsafe_allow_html=True)


def get_model_path(model_type: str) -> str:
    """Get model path based on type"""
    model_mapping = {
        "EfficientNet (Recommended)": "models/deepfake_efficientnet_final.h5",
        "ResNet50": "models/deepfake_resnet_final.h5",
        "Xception": "models/deepfake_xception_final.h5",
        "Custom CNN": "models/deepfake_custom_final.h5"
    }
    return model_mapping.get(model_type, "models/deepfake_efficientnet_final.h5")


if __name__ == '__main__':
    # Initialize session state
    if 'image_result' not in st.session_state:
        st.session_state.image_result = None
    if 'video_result' not in st.session_state:
        st.session_state.video_result = None
    
    main()
