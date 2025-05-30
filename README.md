# Vehicle License Plate Detection System

<img width="1667" alt="Project Screenshot" src="https://github.com/user-attachments/assets/2435c364-803d-4fce-90ce-444f9358fca9" />

## Overview

A Flask-based application for vehicle detection, license plate recognition (ALPR/ANPR), and registered plate management. Processes live camera feeds or video files using YOLOv8 and EasyOCR, with results displayed on a web interface.

---

## Key Features

* Live camera and video file processing.
* Vehicle detection (YOLOv8m).
* Custom license plate detection (YOLO).
* Enhanced OCR for license plates with multiple preprocessing steps.
* SQLite database for plate registration and lookup.
* Web UI for video streaming, controls, and plate management.
* Single image analysis mode.

---

## Tech Stack

* **Backend**: Python, Flask
* **CV & Detection**: OpenCV, Ultralytics YOLOv8
* **OCR**: EasyOCR
* **Database**: SQLite3
* **Frontend**: HTML, CSS, JavaScript

---

## Setup

1.  **Clone repository (if applicable).**
2.  **Create and activate a virtual environment:**
    ```bash
    uv venv # whether you use windows, mac or linux. always uv!
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
4.  **Models:**
    * Ensure `yolov8m.pt` is available.
    * Place your custom license plate models (e.g., `runs/detect/license_plate_detection_n13/weights/best.pt` or `license_plate_detector.pt`) in the project directory or specified paths. Plate detection will be skipped if models are not found.

---

## Run

1.  **Activate virtual environment.**
2.  **Start the application:**
    ```bash
    python app.py
    ```
3.  Open [http://localhost:5001](http://localhost:5001) in your browser.

---

## Core Workflow

1.  **Input**: Video stream from camera or uploaded file.
2.  **Vehicle Detection**: YOLOv8m identifies vehicles in frames.
3.  **License Plate Detection**: A custom YOLO model locates license plates on detected vehicles.
4.  **OCR Processing**: The detected plate region is cropped, enhanced (padding, sharpening, multiple preprocessing techniques like thresholding, CLAHE), and then EasyOCR extracts the plate text.
5.  **Database Check**: The recognized plate text is checked against the SQLite database for registration status.
6.  **Output**: Processed frames with bounding boxes, vehicle type, plate text, and registration status are streamed to the web UI.

---
