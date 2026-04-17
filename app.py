import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from collections import defaultdict
import numpy as np

st.set_page_config(page_title="IPL Tracking", layout="wide")

st.title("🏏 Advanced IPL Player Tracking System")

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi"])

if uploaded_file:
    st.video(uploaded_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("Start Processing"):
        st.info("Processing started...")

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(tfile.name)

        out_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        track_history = defaultdict(list)
        unique_ids = set()

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True)

            for r in results:
                boxes = r.boxes
                if boxes.id is None:
                    continue

                for box, track_id in zip(boxes, boxes.id):
                    x1,y1,x2,y2 = map(int, box.xyxy[0])

                    if int(box.cls[0]) != 0:
                        continue

                    unique_ids.add(int(track_id))

                    if y1 < frame.shape[0]//3:
                        role = "Batsman"
                    elif y1 > 2*frame.shape[0]//3:
                        role = "Bowler"
                    else:
                        role = "Umpire"

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"ID:{int(track_id)} {role}",
                                (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

                    track_history[int(track_id)].append((x1,y1))
                    for i in range(1,len(track_history[int(track_id)])):
                        cv2.line(frame,
                                 track_history[int(track_id)][i-1],
                                 track_history[int(track_id)][i],
                                 (255,0,0),2)

            if out is None:
                h,w,_ = frame.shape
                out = cv2.VideoWriter(out_path,fourcc,20.0,(w,h))

            cv2.putText(frame,f"Players: {len(unique_ids)}",
                        (20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

            out.write(frame)

            current += 1
            progress.progress(min(current/frame_count,1.0))

        cap.release()
        out.release()

        st.success("Done!")
        st.video(out_path)

        with open(out_path,"rb") as f:
            st.download_button("Download Output",f,file_name="output.mp4")
