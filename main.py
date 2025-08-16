import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import supervision as sv
from src import Annotator, ViewTransformer, SpeedEstimator

st.set_page_config(page_title="Vehicle Detection & Speed Estimation", layout="wide")
st.title("🚗 Real-Time Vehicle Detection, Tracking, Counting, and Speed Estimation")

# === CONFIGURATION ===
YOLO_MODEL_PATH = "./models/VisDrone_YOLO_x2.pt"
LINE_Y = 480
WIDTH, HEIGHT = 25, 100
SOURCE_POINTS = np.array([[450, 300], [860, 300], [1900, 720], [-660, 720]])
TARGET_POINTS = np.array([[0, 0], [25, 0], [25, 100], [0, 100]])

# === SESSION STATE ===
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'paused' not in st.session_state:
    st.session_state.paused = False

# === INPUT OPTIONS ===
option = st.radio("Choose input method:", ("📷 Record from Webcam", "📂 Upload Video"))

# --- Webcam Recording ---
if option == "📷 Record from Webcam":
    if not st.session_state.recording:
        if st.button("Start Webcam Recording"):
            st.session_state.recording = True
            st.session_state.video_path = os.path.join(tempfile.gettempdir(), "webcam_video.mp4")
    else:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(st.session_state.video_path, fourcc, 20.0, (640, 480))
        stop_button = st.button("Stop Recording")
        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(display_frame, channels="RGB", use_container_width=True)
            if stop_button:
                st.session_state.recording = False
                break
        cap.release()
        out.release()
        st.success("✅ Webcam recording saved!")

# --- Upload Video ---
elif option == "📂 Upload Video":
    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(uploaded_file.read())
            st.session_state.video_path = temp.name
        st.success("✅ Video uploaded successfully!")

# --- Pause/Resume Button ---
st.button("⏯️ Pause/Resume", on_click=lambda: st.session_state.__setitem__('paused', not st.session_state.paused))

# --- Start Processing ---
start_button = st.button("🚀 Start Vehicle Detection & Tracking")

if start_button:
    video_path = st.session_state.video_path
    if video_path is None:
        st.warning("⚠️ Please upload a video or record from webcam first.")
    else:
        st.info("Loading YOLO model...")
        model = YOLO(YOLO_MODEL_PATH)

        video_info = sv.VideoInfo.from_video_path(video_path)

        # Output video (half resolution)
        output_width = video_info.width // 2
        output_height = video_info.height // 2
        output_video_path = os.path.join(tempfile.gettempdir(), f"processed_{os.path.basename(video_path)}")

        # Tracker and modules
        tracker = sv.ByteTrack(frame_rate=video_info.fps)
        offset = 55
        start, end = sv.Point(offset, LINE_Y), sv.Point(video_info.width - offset, LINE_Y)
        line_zone = sv.LineZone(start, end, minimum_crossing_threshold=1)
        view_transformer = ViewTransformer(np.array(SOURCE_POINTS), np.array(TARGET_POINTS))
        speed_estimator = SpeedEstimator(fps=video_info.fps, view_transformer=view_transformer)
        annotator = Annotator(
            resolution_wh=video_info.resolution_wh,
            box_annotator=True,
            label_annotator=True,
            line_annotator=True,
            multi_class_line_annotator=True,
            trace_annotator=True,
            polygon_zone=SOURCE_POINTS
        )

        stframe = st.empty()
        frame_generator = sv.get_video_frames_generator(video_path)

        with sv.VideoSink(output_video_path, video_info) as sink:
            frame_count = 0
            for frame in frame_generator:
                frame_count += 1

                # Pause handling
                while st.session_state.paused:
                    stframe.text("⏸️ Tracking Paused...")
                    st.time.sleep(0.1)

                # YOLO detection
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)
                line_zone.trigger(detections)
                detections = speed_estimator.update(detections)

                labels = []
                for tracker_id, class_name, speed in zip(detections.tracker_id,
                                                         detections.data["class_name"],
                                                         detections.data["speed"]):
                    text = f"{class_name} #{tracker_id}" if speed == 0 else f"{class_name} {speed}km/h"
                    labels.append(text)

                annotated_frame = annotator.annotate(
                    frame,
                    detections,
                    labels=labels,
                    line_zones=[line_zone],
                    multi_class_zones=[line_zone]
                )

                # --- Smooth live display ---
                if frame_count % 2 == 0:  # display every 2nd frame
                    display_frame = cv2.resize(annotated_frame, (480, 270))
                    stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)

                # --- Resize for smaller output ---
                resized_frame = cv2.resize(annotated_frame, (output_width, output_height))
                sink.write_frame(resized_frame)

        st.success("✅ Processing complete!")
        st.video(output_video_path)
        st.write(f"Total vehicles counted: {line_zone.in_count + line_zone.out_count} "
                 f"(In: {line_zone.in_count}, Out: {line_zone.out_count})")
        st.write(f"YOLO model used: {YOLO_MODEL_PATH}")
