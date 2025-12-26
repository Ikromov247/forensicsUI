"""Streamlit UI for ForensicsUI - Object Finder in Video."""
import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path

from config import Config, OUTPUTS_DIR
from video_processor import VideoProcessor


def main():
    st.set_page_config(
        page_title="ForensicsUI - Object Finder",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 ForensicsUI - Object Finder")
    st.markdown("Find matching objects in video using computer vision")

    # Initialize session state
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "target_features" not in st.session_state:
        st.session_state.target_features = None
    if "target_class" not in st.session_state:
        st.session_state.target_class = None
    if "results" not in st.session_state:
        st.session_state.results = None

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum similarity score to consider a match"
        )
        feature_model = st.selectbox(
            "Feature Model",
            options=["inception", "resnet"],
            help="Model used for feature extraction"
        )
        performance_mode = st.checkbox(
            "Performance Mode",
            value=True,
            help="Extract features less frequently to speed up processing"
        )

    # Create config
    config = Config(
        similarity_threshold=similarity_threshold,
        feature_model=feature_model,
        performance_mode=performance_mode
    )

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.header("Step 1: Upload Target Image")
        target_image = st.file_uploader(
            "Upload an image of the object to find",
            type=["jpg", "jpeg", "png"],
            key="target_upload"
        )

        if target_image:
            # Read and display image
            file_bytes = np.asarray(bytearray(target_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Target Image")

            # Process target
            if st.button("Analyze Target", key="analyze_btn"):
                with st.spinner("Analyzing target image..."):
                    try:
                        if st.session_state.processor is None:
                            st.session_state.processor = VideoProcessor(config)

                        features, cls, bbox = st.session_state.processor.process_target_image(image)
                        st.session_state.target_features = features
                        st.session_state.target_class = cls

                        class_name = st.session_state.processor.detector.class_names.get(cls, str(cls))
                        st.success(f"Target analyzed: **{class_name}** detected")
                    except Exception as e:
                        st.error(f"Error analyzing target: {str(e)}")

    with col2:
        st.header("Step 2: Upload Video")
        video_file = st.file_uploader(
            "Upload video to search",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload"
        )

        if video_file:
            st.video(video_file)

    # Processing section
    st.markdown("---")
    st.header("Step 3: Process Video")

    can_process = (
        st.session_state.target_features is not None and
        video_file is not None
    )

    if not can_process:
        if st.session_state.target_features is None:
            st.info("Please upload and analyze a target image first")
        elif video_file is None:
            st.info("Please upload a video to search")
    else:
        if st.button("🚀 Start Processing", type="primary", key="process_btn"):
            # Save video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name

            progress_bar = st.progress(0, text="Processing video...")
            status_text = st.empty()

            def update_progress(progress: float):
                progress_bar.progress(progress, text=f"Processing: {progress:.0%}")

            try:
                result = st.session_state.processor.process_video(
                    video_path=video_path,
                    target_features=st.session_state.target_features,
                    target_class=st.session_state.target_class,
                    progress_callback=update_progress
                )
                st.session_state.results = result
                progress_bar.progress(1.0, text="Complete!")
                status_text.success(
                    f"Processed {result.total_frames} frames, found {result.total_objects} objects"
                )
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

    # Results section
    if st.session_state.results:
        st.markdown("---")
        st.header("Results")

        result = st.session_state.results

        # Show top matches
        if result.matches:
            st.subheader("Top Matches")
            cols = st.columns(min(5, len(result.matches)))

            for i, match in enumerate(result.matches[:5]):
                with cols[i]:
                    st.metric(
                        label=f"ID: {match.obj_id}",
                        value=f"{match.similarity:.0%}",
                        delta=match.class_name
                    )
                    st.caption(f"Frames: {len(match.frame_ids)}")
        else:
            st.warning("No significant matches found")

        # Show processed video
        st.subheader("Processed Video")
        output_path = Path(result.output_video_path)

        if output_path.exists():
            st.video(str(output_path))

            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name=output_path.name,
                    mime="video/mp4"
                )
        else:
            st.error("Output video not found")

        # Summary stats
        with st.expander("Processing Details"):
            st.write(f"- Total frames processed: {result.total_frames}")
            st.write(f"- Total objects tracked: {result.total_objects}")
            st.write(f"- Matches found: {len(result.matches)}")
            st.write(f"- Output file: {result.output_video_path}")


if __name__ == "__main__":
    main()
