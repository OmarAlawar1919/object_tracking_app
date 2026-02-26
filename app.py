import numpy as np
import cv2
import streamlit as st
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-container {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_frame_delay_seconds(video_fps, playback_speed):
    base_fps = video_fps if video_fps and video_fps > 0 else 30
    delay = 1.0 / (base_fps * playback_speed)
    return max(delay, 0.001)

# Header
st.markdown('<p class="main-header">🎯 Object Tracking Application</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Track moving objects in video using advanced computer vision</p>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    # Detection parameters
    st.subheader("Detection Parameters")
    min_area = st.slider(
        "Minimum Object Area (pixels)",
        min_value=100,
        max_value=2000,
        value=300,
        step=50,
        help="Objects smaller than this area will be ignored"
    )
    
    learning_rate = st.slider(
        "Background Learning Rate",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="How quickly the background model adapts to changes"
    )
    
    st.markdown("---")
    
    # Display options
    st.subheader("Display Options")
    bbox_color = st.color_picker("Bounding Box Color", "#00FF00")
    bbox_thickness = st.slider("Box Thickness", 1, 5, 2)
    show_mask = st.checkbox("Show Detection Mask", value=False)
    show_stats = st.checkbox("Show Real-time Statistics", value=True)
    
    st.markdown("---")
    
    # Playback controls
    st.subheader("Playback Controls")
    playback_speed = st.select_slider(
        "Playback Speed",
        options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        value=1.0,
        format_func=lambda x: f"{x}x"
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Adjust the minimum area to filter out noise and focus on larger objects.")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "📁 Upload your video file",
        type=["mp4", "avi", "mov", "wmv", "mkv"],
        help="Supported formats: MP4, AVI, MOV, WMV, MKV"
    )

if uploaded_file is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    
    if not cap.isOpened():
        st.error("❌ Could not open video file. Please try another file.")
    else:
        # Get video properties
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Display video info
        st.success("✅ Video loaded successfully!")
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("📐 Resolution", f"{width}x{height}")
        with info_col2:
            st.metric("🎬 FPS", f"{fps:.1f}")
        with info_col3:
            st.metric("📊 Total Frames", f"{frame_count}")
        with info_col4:
            st.metric("⏱️ Duration", f"{duration:.1f}s")
        
        st.markdown("---")
        
        # Create columns for video display
        if show_mask:
            video_col1, video_col2 = st.columns(2)
            stframe = video_col1.empty()
            mask_frame = video_col2.empty()
        else:
            stframe = st.empty()
        
        # Statistics placeholders
        if show_stats:
            stats_placeholder = st.empty()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Convert hex color to BGR
        hex_color = bbox_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        
        # Initialize background subtractor
        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16
        )
        
        # Processing variables
        frame_number = 0
        total_detections = 0
        max_objects_frame = 0
        
        start_time = time.time()
        
        # Process video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:            
                break
            
            frame_number += 1
            
            # Apply background subtraction with learning rate
            fg_mask = background_subtractor.apply(frame, learningRate=learning_rate)
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count objects in current frame
            objects_in_frame = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, bbox_thickness)
                    
                    # Add label with area
                    label = f"Area: {int(area)}"
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)
                    
                    objects_in_frame += 1
                    total_detections += 1
            
            max_objects_frame = max(max_objects_frame, objects_in_frame)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_number}/{frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Objects: {objects_in_frame}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frames
            stframe.image(
                convert_color(frame),
                use_container_width=True,
                output_format="JPEG"
            )
            
            if show_mask:
                mask_frame.image(
                    fg_mask,
                    channels="GRAY",
                    use_container_width=True,
                    caption="Detection Mask",
                    output_format="JPEG"
                )
            
            # Update statistics
            if show_stats and frame_number % 10 == 0:  # Update every 10 frames
                elapsed_time = time.time() - start_time
                processing_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                
                with stats_placeholder.container():
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Current Frame", frame_number)
                    with stat_col2:
                        st.metric("Objects Detected", objects_in_frame)
                    with stat_col3:
                        st.metric("Total Detections", total_detections)
                    with stat_col4:
                        st.metric("Processing FPS", f"{processing_fps:.1f}")
            
            # Update progress
            progress = frame_number / frame_count
            progress_bar.progress(progress)
            status_text.text(f"Processing: {progress*100:.1f}% complete")
            
            # Control playback speed
            time.sleep(get_frame_delay_seconds(fps, playback_speed))
        
        cap.release()
        
        # Final statistics
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        st.balloons()
        
        st.markdown("---")
        st.subheader("📈 Processing Summary")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Total Frames Processed", frame_number)
        with summary_col2:
            st.metric("Total Detections", total_detections)
        with summary_col3:
            st.metric("Max Objects in Frame", max_objects_frame)
        with summary_col4:
            avg_objects = total_detections / frame_number if frame_number > 0 else 0
            st.metric("Avg Detections/Frame", f"{avg_objects:.2f}")
        
        st.success("🎉 Video processing completed successfully!")
        
else:
    # Show instructions when no file is uploaded
    st.info("""
    ### 📋 How to use this application:
    
    1. **Upload a video** using the file uploader above
    2. **Adjust settings** in the sidebar to fine-tune detection
    3. **Watch** as objects are tracked in real-time
    4. **Review statistics** after processing completes
    
    ### 🔍 What this app does:
    
    - Detects moving objects using background subtraction
    - Draws bounding boxes around detected objects
    - Provides real-time statistics and metrics
    - Allows customization of detection parameters
    
    ### ⚡ Tips for best results:
    
    - Use videos with a relatively static background
    - Adjust minimum area to filter out small movements
    - Increase learning rate for dynamic backgrounds
    - Decrease learning rate for static backgrounds
    """)