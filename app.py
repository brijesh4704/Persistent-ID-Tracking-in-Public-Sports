import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from collections import defaultdict

st.set_page_config(page_title="Fast AI Tracking", layout="wide")

st.title("⚡ Fast AI Player Tracking Dashboard")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
frame_skip = st.sidebar.slider("Frame Skip", 2, 6, 4)  # increased
max_frames = st.sidebar.slider("Max Frames", 100, 400, 200)

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4","avi"])

if uploaded_file:
    st.video(uploaded_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("🚀 Start Fast Processing"):

        with st.spinner("⚡ Optimizing & Processing..."):
            model = load_model()
            cap = cv2.VideoCapture(tfile.name)

            out_path = "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None

            unique_ids = set()
            frame_count = 0

            progress = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 🔥 Skip more frames (big speed boost)
                if frame_count % frame_skip != 0:
                    continue

                # 🔥 Limit processing
                if frame_count > max_frames:
                    break

                # 🔥 Smaller resolution
                frame = cv2.resize(frame, (480, 270))

                # 🔥 Faster tracking settings
                results = model.track(
                    frame,
                    persist=True,
                    conf=0.4,     # lower confidence → faster
                    iou=0.5
                )

                for r in results:
                    if r.boxes.id is None:
                        continue

                    for box, track_id in zip(r.boxes, r.boxes.id):
                        if int(box.cls[0]) != 0:
                            continue

                        unique_ids.add(int(track_id))

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame, f"ID:{int(track_id)}",
                                    (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,
                                    (255,255,255),2)

                if out is None:
                    h, w, _ = frame.shape
                    out = cv2.VideoWriter(out_path, fourcc, 10.0, (w,h))

                out.write(frame)

                progress.progress(min(frame_count/max_frames,1.0))

            cap.release()
            out.release()

        st.success("✅ Done Fast!")

        col1, col2 = st.columns(2)
        col1.metric("👥 Players", len(unique_ids))
        col2.metric("🎞 Frames", frame_count)

        st.video(out_path)

        with open(out_path,"rb") as f:
            st.download_button("⬇️ Download", f, file_name="output.mp4")
