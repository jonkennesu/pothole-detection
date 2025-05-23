import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

import streamlit as st
import cv2
import tempfile
import numpy as np
import hashlib
from PIL import Image
import io

@st.cache_resource
def load_model(path):
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

def compute_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def process_image(image, model, threshold):
    """Process a single image and return annotated result with pothole count"""
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img_rgb, conf=threshold, verbose=False)
        
        # Count potholes
        pothole_count = 0
        if results[0].boxes is not None:
            pothole_count = len(results[0].boxes)
        
        # Get annotated image
        annotated_frame = results[0].plot()
        
        # Add pothole count text to the annotated image
        text = f"Potholes: {pothole_count}"
        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, pothole_count
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, 0

# Page configuration
st.set_page_config(page_title="Pothole Detection", layout="wide")
st.title("🕳️ Pothole Detection System")
st.markdown("Upload an image or video to detect potholes automatically")

# Initialize session state
if 'processed_hash' not in st.session_state:
    st.session_state.processed_hash = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'annotated_video_path' not in st.session_state:
    st.session_state.annotated_video_path = None
if 'original_video_path' not in st.session_state:
    st.session_state.original_video_path = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []
if 'annotated_frames' not in st.session_state:
    st.session_state.annotated_frames = []

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image or video file", 
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    help="Supported formats: JPG, JPEG, PNG for images; MP4, MOV, AVI for videos"
)

# Threshold slider
threshold = st.slider(
    "Detection Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Lower values detect more objects but may include false positives"
)

if uploaded_file is not None:
    # Get file hash and type
    file_bytes = uploaded_file.read()
    current_hash = compute_file_hash(file_bytes)
    file_extension = uploaded_file.name.split('.')[-1].lower()
    is_image = file_extension in ['jpg', 'jpeg', 'png']
    is_video = file_extension in ['mp4', 'mov', 'avi']
    
    # Check if we need to run inference (new file uploaded)
    need_inference = current_hash != st.session_state.processed_hash
    
    if need_inference:
        st.session_state.processed_hash = current_hash
        st.session_state.file_type = 'image' if is_image else 'video'
        st.session_state.inference_done = False
        
        # Load model
        try:
            model_path = "pothole_best.pt"
            if not os.path.exists(model_path):
                st.error(f"❌ Model file '{model_path}' not found!")
                st.info("📋 Please ensure 'pothole_best.pt' model file is in the same directory as this script")
                st.stop()
            
            with st.spinner("🔄 Loading AI model..."):
                model = load_model(model_path)
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
        
        if is_image:
            # Process image
            st.info("🔍 Processing image...")
            
            try:
                # Convert bytes to image
                image = Image.open(io.BytesIO(file_bytes))
                image_np = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_np.shape) == 3:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image_np
                
                # Store original image
                st.session_state.original_image = image_np
                
                # Process with model
                result = process_image(image_bgr, model, threshold)
                if result[0] is not None:
                    annotated_img, pothole_count = result
                    st.session_state.annotated_image = annotated_img
                    st.session_state.inference_done = True
                    st.success(f"✅ Processing complete! Found {pothole_count} potholes")
                else:
                    st.error("❌ Failed to process image")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                st.stop()
            
        elif is_video:
            # Process video
            st.info("🎥 Processing video... This may take a while.")
            
            try:
                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                    tmp.write(file_bytes)
                    temp_video_path = tmp.name
                
                st.session_state.original_video_path = temp_video_path
                
                # Open video
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    st.error("❌ Could not open video file")
                    st.stop()
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Output video path
                output_path = os.path.join(tempfile.gettempdir(), f"annotated_{current_hash}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                # Create placeholders for real-time display
                progress_bar = st.progress(0)
                status_text = st.empty()
                video_placeholder = st.empty()
                
                frame_count = 0
                total_potholes = 0
                original_frames = []
                annotated_frames_list = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Store original frame (convert BGR to RGB for display)
                    original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_frames.append(original_frame_rgb)
                    
                    # Process frame
                    result = process_image(frame, model, threshold)
                    if result[0] is not None:
                        annotated_frame, pothole_count = result
                        total_potholes += pothole_count
                        annotated_frames_list.append(annotated_frame)
                        
                        # Write to output video
                        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                        
                        # Update display every 30 frames
                        if frame_count % max(1, int(fps)) == 0:
                            video_placeholder.image(annotated_frame, channels="RGB", width=600)
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} - Potholes detected so far: {total_potholes}")
                
                cap.release()
                out.release()
                
                # Store frames in session state
                st.session_state.video_frames = original_frames
                st.session_state.annotated_frames = annotated_frames_list
                
                st.session_state.annotated_video_path = output_path
                st.session_state.inference_done = True
                
                # Clear the real-time display elements
                video_placeholder.empty()
                
                st.success(f"✅ Video processing complete! Total potholes detected: {total_potholes}")
                
            except Exception as e:
                st.error(f"❌ Error processing video: {e}")
                st.stop()
    
    # Display results
    if st.session_state.inference_done:
        if st.session_state.file_type == 'image' and st.session_state.original_image is not None:
            # For images: show side by side comparison
            st.subheader("🖼️ Detection Results")
            
            # If threshold changed, reprocess the image
            if not need_inference:
                try:
                    model = load_model("pothole_best.pt")
                    image_bgr = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_RGB2BGR)
                    result = process_image(image_bgr, model, threshold)
                    if result[0] is not None:
                        annotated_img, pothole_count = result
                        st.session_state.annotated_image = annotated_img
                except Exception as e:
                    st.error(f"Error reprocessing image: {e}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(st.session_state.original_image, width=400)
            
            with col2:
                st.markdown("**Detected Potholes**")
                st.image(st.session_state.annotated_image, channels="RGB", width=400)
            
            # Download button for annotated image
            if st.session_state.annotated_image is not None:
                pil_image = Image.fromarray(st.session_state.annotated_image)
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buf.getvalue(),
                    file_name=f"annotated_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )
        
        elif st.session_state.file_type == 'video':
            # For videos: show frame selector and side-by-side comparison
            st.subheader("🎬 Video Frame Analysis")
            
            if st.session_state.video_frames and st.session_state.annotated_frames:
                total_frames = len(st.session_state.video_frames)
                
                # Frame selector slider
                selected_frame = st.slider(
                    "Select Frame to Display",
                    min_value=0,
                    max_value=total_frames - 1,
                    value=0,
                    help=f"Choose which frame to view (Total frames: {total_frames})"
                )
                
                # Display selected frame side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Frame**")
                    st.image(st.session_state.video_frames[selected_frame], width=400)
                
                with col2:
                    st.markdown("**Detected Potholes**")
                    st.image(st.session_state.annotated_frames[selected_frame], channels="RGB", width=400)
                
                # Show frame info
                st.info(f"Displaying frame {selected_frame + 1} of {total_frames}")
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if os.path.exists(st.session_state.original_video_path):
                        with open(st.session_state.original_video_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Original Video",
                                data=f.read(),
                                file_name=f"original_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                
                with col2:
                    if os.path.exists(st.session_state.annotated_video_path):
                        with open(st.session_state.annotated_video_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Annotated Video",
                                data=f.read(),
                                file_name=f"annotated_{uploaded_file.name}",
                                mime="video/mp4"
                            )
            else:
                st.error("Video frames not found. Please try uploading the video again.")

else:
    st.info("👆 Please upload an image or video file to get started")
    st.markdown("""
    ### Features:
    - 🖼️ **Image Detection**: Upload JPG, JPEG, or PNG images
    - 🎥 **Video Detection**: Upload MP4, MOV, or AVI videos  
    - 🎚️ **Adjustable Threshold**: Fine-tune detection sensitivity
    - 📊 **Real-time Processing**: Watch video analysis in progress
    - 📥 **Download Results**: Get both original and annotated files
    - 🔄 **Smart Caching**: Avoid reprocessing the same file
    """)

# Add footer
st.markdown("---")
st.markdown("*Make sure you have the 'pothole_best.pt' model file in your working directory*")
