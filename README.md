# 🏏 Multi-Object Detection and Tracking in Sports Video

## 📌 Objective

This project implements a computer vision pipeline for detecting and tracking multiple objects in sports video footage with persistent ID assignment.

---

## 🎥 Dataset / Video Source

Public cricket video used:
https://youtube.com/shorts/iM0MBL6jpLQ?si=DSycbjrfA6oOw5Qm

---

## 🚀 Features

* Multi-object detection using YOLOv8
* Persistent ID tracking using ByteTrack
* Role classification (Batsman, Bowler, Umpire)
* Trajectory visualization
* Annotated output video generation
* Streamlit web interface for interaction

---

## 🧠 Technologies Used

* Python
* OpenCV
* Ultralytics YOLOv8
* ByteTrack (built-in with YOLO tracking)
* Streamlit

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
streamlit run app.py
```

---

## 📊 Pipeline Overview

1. Input video is loaded
2. YOLOv8 detects objects (persons)
3. ByteTrack assigns unique IDs
4. IDs are maintained across frames
5. Bounding boxes + IDs are drawn
6. Trajectories are visualized
7. Output video is generated

---

## ⚠️ Assumptions

* All players are detected as "person" class
* Role classification is based on position heuristics

---

## ❌ Limitations

* Role classification is not fully accurate
* Performance drops with heavy occlusion
* No team classification

---

## 🚀 Future Improvements

* Team classification using color clustering
* Speed estimation
* Heatmaps and analytics
* Real-time deployment

---

## 📸 Output
asset folder

---

## 🎯 Conclusion

The system successfully performs multi-object detection and tracking with persistent IDs in sports video.
