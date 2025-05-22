import streamlit as st
import cv2
import tempfile
import numpy as np
import os
import hashlib
from ultralytics import YOLO
from PIL import Image
import io

@st.cache_resource
def load_model(path):
    return YOLO(path)

def compute_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def process_image(image, model, threshold):
    """Process a single image and return annotated result with pothole count"""
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
if 'inference_done' not in st.session_state:
    st.session_state.inference_done = False

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
            model = load_model("pothole_best.pt")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Make sure 'best.pt' model file is in the same directory as this script")
            st.stop()
        
        if is_image:
            # Process image
            st.info("🔍 Processing image...")
            
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
            annotated_img, pothole_count = process_image(image_bgr, model, threshold)
            st.session_state.annotated_image = annotated_img
            st.session_state.inference_done = True
            
            st.success(f"✅ Processing complete! Found {pothole_count} potholes")
            
        elif is_video:
            # Process video
            st.info("🎥 Processing video... This may take a while.")
            
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                tmp.write(file_bytes)
                temp_video_path = tmp.name
            
            st.session_state.original_video_path = temp_video_path
            
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, pothole_count = process_image(frame, model, threshold)
                total_potholes += pothole_count
                
                # Write to output video
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                # Update display every 30 frames (roughly every second for 30fps video)
                if frame_count % max(1, int(fps)) == 0:
                    video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} - Potholes detected so far: {total_potholes}")
            
            cap.release()
            out.release()
            
            st.session_state.annotated_video_path = output_path
            st.session_state.inference_done = True
            
            # Clear the real-time display elements
            video_placeholder.empty()
            
            st.success(f"✅ Video processing complete! Total potholes detected: {total_potholes}")
    
    # Display results based on file type and threshold changes
    if st.session_state.inference_done:
        if st.session_state.file_type == 'image' and st.session_state.original_image is not None:
            # For images: show side by side comparison
            st.subheader("🖼️ Detection Results")
            
            # If threshold changed, reprocess the image
            if not need_inference:  # Only reprocess if file hasn't changed
                try:
                    model = load_model("best.pt")
                    # Convert original image for processing
                    image_bgr = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_RGB2BGR)
                    annotated_img, pothole_count = process_image(image_bgr, model, threshold)
                    st.session_state.annotated_image = annotated_img
                except Exception as e:
                    st.error(f"Error reprocessing image: {e}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(st.session_state.original_image, use_container_width=True)
            
            with col2:
                st.markdown("**Detected Potholes**")
                st.image(st.session_state.annotated_image, channels="RGB", use_container_width=True)
            
            # Download button for annotated image
            if st.session_state.annotated_image is not None:
                # Convert to PIL Image for download
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
            # For videos: show annotated video
            st.subheader("🎬 Annotated Video")
            
            if st.session_state.annotated_video_path and os.path.exists(st.session_state.annotated_video_path):
                # Display the processed video
                video_file = open(st.session_state.annotated_video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                video_file.close()
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Original Video",
                        data=open(st.session_state.original_video_path, "rb").read(),
                        file_name=f"original_{uploaded_file.name}",
                        mime="video/mp4"
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=video_bytes,
                        file_name=f"annotated_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            else:
                st.error("Annotated video not found. Please try uploading the video again.")

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
st.markdown("*Make sure you have the 'best.pt' model file in your working directory*")
