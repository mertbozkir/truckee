from flask import Flask, render_template, request, jsonify, Response
import cv2
import easyocr
from ultralytics import YOLO
import sqlite3
import os
import threading
import time
import numpy as np
import base64
import PIL.Image
import string

try:
    PIL.Image.ANTIALIAS
except AttributeError:  # Pillow 10.0.0 and later
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

app = Flask(__name__)

# Initialize models
print("🔄 Loading YOLO vehicle detection model...")
yolo_model = YOLO("yolov8m.pt").half()

print("🔄 Loading license plate detection model...")
try:
    # Try the most recent trained model first (n13)
    license_plate_detector = YOLO('runs/detect/license_plate_detection_n13/weights/best.pt')
    print("✅ Custom license plate detector (n13 - latest) loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load n13 model: {e}")
    print("🔄 Trying n12 model...")
    try:
        license_plate_detector = YOLO('runs/detect/license_plate_detection_n12/weights/best.pt')
        print("✅ Custom license plate detector (n12) loaded successfully!")
    except Exception as e2:
        print(f"❌ Failed to load n12 model: {e2}")
        print("🔄 Trying fallback model...")
        try:
            license_plate_detector = YOLO('license_plate_detector.pt')
            print("✅ Fallback license plate detector loaded successfully!")
        except Exception as e3:
            print(f"❌ Failed to load fallback license plate detector: {e3}")
            print("📥 Please ensure you have a trained license plate detection model")
            license_plate_detector = None

print("🔄 Loading EasyOCR...")
ocr_reader = easyocr.Reader(["en"])

print("✅ Models loaded successfully!")


def read_license_plate(license_plate_crop):
    """Read the license plate text from the given cropped image."""
    detections = ocr_reader.readtext(license_plate_crop)
    
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(" ", "")

        # For non-standard formats, use a more flexible approach
        # Clean the text and check if it has reasonable plate characteristics
        cleaned_text = "".join(c for c in text if c.isalnum())
        if (
            len(cleaned_text) >= 4
            and any(c.isdigit() for c in cleaned_text)
            and any(c.isalpha() for c in cleaned_text)
        ):
            return cleaned_text.lower(), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """Retrieve the vehicle coordinates and ID based on the license plate coordinates."""
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


# Initialize database
def init_db():
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            plate_number TEXT PRIMARY KEY,
            vehicle_type TEXT NOT NULL DEFAULT 'truck'
        )
    """)
    conn.commit()
    conn.close()


init_db()


class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.is_processing = False
        self.processing_status = "idle"
        self.frame_count = 0
        self.total_frames = 0
        # Add tracking variables for consistency
        self.recent_detections = {}  # Store recent plate detections by vehicle area
        self.detection_history = []  # Keep history of detections
        self.detection_cache_size = 10  # Number of frames to keep in history

    def start_camera(self, camera_index=0):
        self.processing_status = "initializing"

        if self.cap:
            self.cap.release()

        import os

        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

        print(f"🔄 Initializing camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)

        if self.cap.isOpened():
            # Set camera properties for faster processing
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Test if we can read a frame
            ret, frame = self.cap.read()
            if ret:
                print(f"✅ Camera {camera_index} initialized successfully")
                self.processing_status = "ready"
                self.is_processing = True
                self.frame_count = 0
                return True
            else:
                self.cap.release()
                self.processing_status = "error"

        print(f"❌ Camera {camera_index} not found")
        self.processing_status = "error"
        return False

    def get_available_cameras(self):
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(i)
                cap.release()
        return cameras

    def start_video(self, video_path):
        self.processing_status = "loading_video"
        print(f"🔄 Loading video: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            self.processing_status = "error"
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0

        print(f"✅ Video loaded - {self.total_frames} frames")
        self.processing_status = "processing_video"
        self.is_processing = True
        return True

    def stop(self):
        self.is_processing = False
        if self.cap:
            self.cap.release()

    def check_plate_registered(self, plate_text):
        if not plate_text:
            return False
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT plate_number FROM plates WHERE plate_number = ?", (plate_text,)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def get_vehicle_key(self, bbox):
        """Generate a key for vehicle based on its approximate position"""
        x1, y1, x2, y2 = bbox[:4]
        # Use center point and size to create a rough vehicle identifier
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        # Round to grid for slight position variations
        grid_size = 50
        grid_x = (center_x // grid_size) * grid_size
        grid_y = (center_y // grid_size) * grid_size
        return f"{grid_x}_{grid_y}_{width//10}_{height//10}"

    def find_best_plate_for_vehicle(self, vehicle_key, current_frame):
        """Find the best plate detection for a vehicle using recent history"""
        if vehicle_key not in self.recent_detections:
            return None

        # Get recent detections for this vehicle
        vehicle_detections = self.recent_detections[vehicle_key]

        # Filter detections from recent frames (within last 15 frames)
        recent_detections = [
            d for d in vehicle_detections if current_frame - d["frame"] <= 15
        ]

        if not recent_detections:
            return None

        # Find the detection with highest confidence
        best_detection = max(recent_detections, key=lambda x: x["confidence"])

        # Only use if confidence is reasonable and not too old
        if (
            best_detection["confidence"] > 0.3
            and (current_frame - best_detection["frame"]) <= 10
        ):
            return best_detection["plate_text"]

        return None

    def add_detection_to_history(self, vehicle_key, plate_text, confidence, frame_num):
        """Add a detection to the history for temporal consistency"""
        if vehicle_key not in self.recent_detections:
            self.recent_detections[vehicle_key] = []

        detection = {
            "plate_text": plate_text,
            "confidence": confidence,
            "frame": frame_num,
        }

        self.recent_detections[vehicle_key].append(detection)

        # Keep only recent detections (last 20 frames worth)
        self.recent_detections[vehicle_key] = [
            d
            for d in self.recent_detections[vehicle_key]
            if frame_num - d["frame"] <= 20
        ]

    def process_frame(self, frame):
        import time

        start_time = time.time()

        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame.copy()

        # Vehicle detection with YOLO
        vehicle_results = yolo_model(
            frame_resized, verbose=False, imgsz=640, conf=0.4, device="mps"
        )

        # Prepare vehicle detections for tracking format
        vehicle_detections = []
        vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        for r in vehicle_results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls in vehicles and conf > 0.4:
                        # Scale coordinates back to original frame
                        if width > 640:
                            scale_back = width / 640
                            x1, y1, x2, y2 = (box.xyxy[0] * scale_back).int().tolist()
                        else:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                        vehicle_detections.append([x1, y1, x2, y2, conf, cls])

        detected_vehicles = []

        # License plate detection - Process more frequently for better consistency
        process_plates = True  # Process every frame for better detection

        if license_plate_detector is not None and process_plates:
            plate_results = license_plate_detector(
                frame, verbose=False, conf=0.1  # Much lower confidence threshold
            )

            print(f"Frame {self.frame_count}: Found {len(plate_results[0].boxes) if plate_results[0].boxes is not None else 0} potential plates")

            for plate_result in plate_results:
                if plate_result.boxes is not None:
                    for plate_box in plate_result.boxes.data.tolist():
                        px1, py1, px2, py2, plate_score, plate_class_id = plate_box
                        
                        print(f"  Plate detected at ({px1:.0f},{py1:.0f},{px2:.0f},{py2:.0f}) with confidence {plate_score:.3f}")

                        # Find which vehicle this plate belongs to
                        best_vehicle = None
                        best_overlap = 0

                        for vehicle in vehicle_detections:
                            vx1, vy1, vx2, vy2, v_conf, v_cls = vehicle

                            # Check if plate is inside vehicle bbox with some tolerance
                            tolerance = 20  # Increased tolerance
                            if (
                                px1 > vx1 - tolerance
                                and py1 > vy1 - tolerance
                                and px2 < vx2 + tolerance
                                and py2 < vy2 + tolerance
                            ):
                                # Calculate overlap area
                                overlap_area = (px2 - px1) * (py2 - py1)
                                if overlap_area > best_overlap:
                                    best_overlap = overlap_area
                                    best_vehicle = vehicle

                        if best_vehicle is not None:
                            vx1, vy1, vx2, vy2, v_conf, v_cls = best_vehicle
                            vehicle_key = self.get_vehicle_key([vx1, vy1, vx2, vy2])
                            
                            print(f"    Matched to vehicle at ({vx1:.0f},{vy1:.0f},{vx2:.0f},{vy2:.0f})")

                            # Draw license plate bounding box for debugging
                            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 255), 2)

                            # Crop license plate for OCR
                            plate_crop = frame[
                                int(py1) : int(py2), int(px1) : int(px2), :
                            ]

                            plate_text = ""
                            ocr_confidence = 0

                            if plate_crop.size > 0:
                                try:
                                    print(f"    Processing plate crop: {plate_crop.shape}")
                                    
                                    # Enhance the plate crop with padding
                                    enhanced_crop = enhance_plate_crop(plate_crop, padding=10)
                                    
                                    # Save original crop for debugging
                                    debug_orig_path = f"debug_plates/orig_frame_{self.frame_count}.jpg"
                                    os.makedirs("debug_plates", exist_ok=True)
                                    cv2.imwrite(debug_orig_path, plate_crop)
                                    cv2.imwrite(f"debug_plates/enhanced_frame_{self.frame_count}.jpg", enhanced_crop)
                                    
                                    # Use enhanced OCR
                                    plate_text, ocr_confidence = read_license_plate_enhanced(enhanced_crop)
                                    
                                    print(f"    OCR result: '{plate_text}' (confidence: {ocr_confidence})")

                                    # Lower the confidence threshold significantly for testing
                                    if plate_text and ocr_confidence and ocr_confidence > 0.1:  # Much lower threshold
                                        print(f"    ACCEPTED: '{plate_text}' with confidence {ocr_confidence}")
                                        # Add to detection history
                                        self.add_detection_to_history(
                                            vehicle_key,
                                            plate_text,
                                            ocr_confidence * plate_score,
                                            self.frame_count,
                                        )
                                    else:
                                        print(f"    REJECTED: confidence too low or no text")
                                            
                                except Exception as e:
                                    print(f"    OCR error: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # Remove this vehicle from the list to avoid duplicates
                            if best_vehicle in vehicle_detections:
                                vehicle_detections.remove(best_vehicle)
                        else:
                            print(f"    No matching vehicle found")
                            # Still draw the plate box for debugging
                            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 255), 2)

        # Process all vehicles (including those without fresh plate detections)
        for vehicle in vehicle_detections:
            vx1, vy1, vx2, vy2, v_conf, v_cls = vehicle
            vehicle_key = self.get_vehicle_key([vx1, vy1, vx2, vy2])

            # Try to get a recent plate detection for this vehicle
            plate_text = self.find_best_plate_for_vehicle(vehicle_key, self.frame_count)

            # Determine vehicle type
            if v_cls == 7:
                vehicle_type = "TRUCK"
            else:
                vehicle_type = "OTHER"

            # Check if plate is registered
            is_registered = (
                self.check_plate_registered(plate_text) if plate_text else False
            )

            # Set colors and status
            if v_cls == 7:  # Truck
                if is_registered:
                    color = (0, 255, 0)  # Green for registered trucks
                    status = "✓ REGISTERED TRUCK"
                else:
                    color = (0, 165, 255)  # Orange for trucks
                    status = "TRUCK" + (
                        " - UNREGISTERED" if plate_text else " - NO PLATE"
                    )
            else:  # Other vehicles
                color = (0, 0, 255)  # Red for all other vehicles
                status = "CAR" + (" - UNREGISTERED" if plate_text else " - NO PLATE")

            # Draw vehicle bounding box
            cv2.rectangle(frame, (int(vx1), int(vy1)), (int(vx2), int(vy2)), color, 2)

            # Draw label
            label = f"{v_conf:.1f}"
            if plate_text:
                label += f" {plate_text}"
            label = f"{status} ({label})"

            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(
                frame,
                (int(vx1), int(vy1) - label_size[1] - 8),
                (int(vx1) + label_size[0] + 4, int(vy1)),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (int(vx1) + 2, int(vy1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            detected_vehicles.append(
                {
                    "type": vehicle_type,
                    "plate": plate_text or "",
                    "bbox": [int(vx1), int(vy1), int(vx2), int(vy2)],
                    "confidence": v_conf,
                    "registered": is_registered,
                }
            )

        # Clean up old detections periodically
        if self.frame_count % 50 == 0:  # Every 50 frames
            self.cleanup_old_detections()

        # Add processing info overlay
        process_time = (time.time() - start_time) * 1000
        fps = 1000 / process_time if process_time > 0 else 0

        info_text = f"FPS: {fps:.1f} | Process: {process_time:.1f}ms | Status: {self.processing_status}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        if self.total_frames > 0:
            progress = (self.frame_count / self.total_frames) * 100
            progress_text = (
                f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames})"
            )
            cv2.putText(
                frame,
                progress_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                progress_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

        # Display detection cache info for debugging
        if len(self.recent_detections) > 0:
            cache_text = f"Tracking: {len(self.recent_detections)} vehicles"
            cv2.putText(
                frame,
                cache_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                cache_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        self.frame_count += 1
        return frame, detected_vehicles

    def cleanup_old_detections(self):
        """Remove old detections from memory"""
        current_frame = self.frame_count
        for vehicle_key in list(self.recent_detections.keys()):
            # Remove vehicles that haven't been seen in the last 30 frames
            self.recent_detections[vehicle_key] = [
                d
                for d in self.recent_detections[vehicle_key]
                if current_frame - d["frame"] <= 30
            ]
            # Remove vehicle key if no recent detections
            if not self.recent_detections[vehicle_key]:
                del self.recent_detections[vehicle_key]

    def generate_frames(self):
        print("🎬 Starting frame generation...")

        while self.is_processing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.processing_status == "processing_video":
                    print("✅ Video processing completed")
                    self.processing_status = "completed"
                break

            processed_frame, detections = self.process_frame(frame)

            ret, buffer = cv2.imencode(
                ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            if self.processing_status == "processing_video":
                time.sleep(0.03)

        print("🛑 Frame generation stopped")
        self.processing_status = "idle"


video_processor = VideoProcessor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        video_processor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/get_cameras")
def get_cameras():
    cameras = video_processor.get_available_cameras()
    camera_list = []
    for i in cameras:
        if i == 0:
            camera_list.append({"index": i, "name": "Default Camera"})
        else:
            camera_list.append({"index": i, "name": f"Camera {i} (USB/Mobile)"})
    return jsonify(camera_list)


@app.route("/get_status")
def get_status():
    progress = 0
    if video_processor.total_frames > 0:
        progress = (video_processor.frame_count / video_processor.total_frames) * 100

    return jsonify(
        {
            "status": video_processor.processing_status,
            "frame_count": video_processor.frame_count,
            "total_frames": video_processor.total_frames,
            "progress": progress,
        }
    )


@app.route("/start_camera", methods=["POST"])
def start_camera():
    data = request.get_json()
    camera_index = data.get("camera_index", 0)
    success = video_processor.start_camera(camera_index)
    if success:
        return jsonify(
            {
                "status": f"Camera {camera_index} started",
                "processing_status": video_processor.processing_status,
            }
        )
    else:
        return jsonify(
            {"error": f"Camera {camera_index} not accessible. Check permissions."}
        ), 400


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"})

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "No video selected"})

    video_path = "uploads/" + video.filename
    os.makedirs("uploads", exist_ok=True)
    video.save(video_path)

    success = video_processor.start_video(video_path)
    if success:
        return jsonify({"status": "Video processing started"})
    else:
        return jsonify({"error": "Failed to open video file"}), 400


@app.route("/stop_processing", methods=["POST"])
def stop_processing():
    video_processor.stop()
    return jsonify({"status": "Processing stopped"})


@app.route("/test_image", methods=["POST"])
def test_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"})

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No image selected"})

    try:
        # Read and decode image
        image_data = image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"})

        display_img = img.copy()

        # Vehicle detection
        vehicle_results = yolo_model(img, conf=0.4, verbose=False)

        # Prepare vehicle detections
        vehicle_detections = []
        vehicles = [2, 3, 5, 7]

        for r in vehicle_results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls in vehicles:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        vehicle_detections.append([x1, y1, x2, y2, conf, cls])

        detections = []

        # License plate detection (only if model is available)
        if license_plate_detector is not None:
            plate_results = license_plate_detector(img, verbose=False)

            for plate_result in plate_results:
                if plate_result.boxes is not None:
                    for plate_box in plate_result.boxes.data.tolist():
                        px1, py1, px2, py2, plate_score, plate_class_id = plate_box

                        # Find which vehicle this plate belongs to
                        best_vehicle = None
                        best_overlap = 0

                        for vehicle in vehicle_detections:
                            vx1, vy1, vx2, vy2, v_conf, v_cls = vehicle

                            if px1 > vx1 and py1 > vy1 and px2 < vx2 and py2 < vy2:
                                overlap_area = (px2 - px1) * (py2 - py1)
                                if overlap_area > best_overlap:
                                    best_overlap = overlap_area
                                    best_vehicle = vehicle

                        if best_vehicle is not None:
                            vx1, vy1, vx2, vy2, v_conf, v_cls = best_vehicle

                            # Crop license plate for OCR
                            plate_crop = img[
                                int(py1) : int(py2), int(px1) : int(px2), :
                            ]

                            plate_text = ""
                            if plate_crop.size > 0:
                                try:
                                    plate_crop_gray = cv2.cvtColor(
                                        plate_crop, cv2.COLOR_BGR2GRAY
                                    )
                                    _, plate_crop_thresh = cv2.threshold(
                                        plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                                    )
                                    plate_text, plate_text_score = read_license_plate(
                                        plate_crop_thresh
                                    )
                                except Exception as e:
                                    print(f"OCR Error during test_image: {e}")
                                    pass

                            vehicle_type = "truck" if v_cls == 7 else "car"

                            is_registered = False
                            if plate_text:
                                conn = sqlite3.connect("data.db")
                                cursor = conn.cursor()
                                cursor.execute(
                                    "SELECT 1 FROM plates WHERE plate_number = ?",
                                    (plate_text,),
                                )
                                is_registered = cursor.fetchone() is not None
                                conn.close()

                            draw_color_tuple = (0, 0, 255)  # Default Red
                            status_text = "CAR"
                            if vehicle_type == "truck":
                                if is_registered:
                                    status_text = "ALLOWED TRUCK"
                                    draw_color_tuple = (0, 255, 0)  # Green
                                else:
                                    status_text = "TRUCK"
                                    draw_color_tuple = (0, 165, 255)  # Orange

                            # Draw vehicle bounding box
                            cv2.rectangle(
                                display_img,
                                (int(vx1), int(vy1)),
                                (int(vx2), int(vy2)),
                                draw_color_tuple,
                                3,
                            )

                            # Draw license plate bounding box
                            cv2.rectangle(
                                display_img,
                                (int(px1), int(py1)),
                                (int(px2), int(py2)),
                                (0, 255, 255),
                                2,
                            )

                            # Prepare label text lines
                            label_lines = [
                                f"{status_text}",
                                f"Conf: {(v_conf * 100):.1f}%",
                                f"Plate: {plate_text or 'No Plate'}",
                            ]

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            thickness = 2
                            text_color = (255, 255, 255)

                            # Calculate overall label background size
                            max_line_width = 0
                            total_text_height = 0
                            line_heights_and_baselines = []

                            for line in label_lines:
                                (lw, lh), baseline = cv2.getTextSize(
                                    line, font, font_scale, thickness
                                )
                                max_line_width = max(max_line_width, lw)
                                line_height_with_baseline = lh + baseline
                                line_heights_and_baselines.append(
                                    line_height_with_baseline
                                )
                                total_text_height += lh + 5

                            # Position background rectangle above the bounding box
                            bg_y2 = vy1 - 5
                            bg_y1 = bg_y2 - total_text_height - 5
                            bg_x1 = vx1
                            bg_x2 = vx1 + max_line_width + 10

                            bg_y1 = max(0, bg_y1)

                            cv2.rectangle(
                                display_img,
                                (bg_x1, bg_y1),
                                (bg_x2, bg_y2),
                                draw_color_tuple,
                                -1,
                            )

                            # Draw text lines
                            current_text_y = (
                                bg_y1
                                + line_heights_and_baselines[0]
                                - (
                                    line_heights_and_baselines[0]
                                    - cv2.getTextSize(
                                        label_lines[0], font, font_scale, thickness
                                    )[0][1]
                                )
                                + 2
                            )

                            for i, line in enumerate(label_lines):
                                cv2.putText(
                                    display_img,
                                    line,
                                    (vx1 + 5, current_text_y),
                                    font,
                                    font_scale,
                                    text_color,
                                    thickness,
                                    cv2.LINE_AA,
                                )
                                if i < len(label_lines) - 1:
                                    current_text_y += (
                                        cv2.getTextSize(
                                            label_lines[i + 1],
                                            font,
                                            font_scale,
                                            thickness,
                                        )[0][1]
                                        + 5
                                    )

                            detections.append(
                                {
                                    "type": vehicle_type,
                                    "plate": plate_text or "no plate",
                                    "confidence": v_conf,
                                    "color": "red"
                                    if vehicle_type == "car"
                                    else "orange"
                                    if vehicle_type == "truck" and not is_registered
                                    else "green",
                                    "status": status_text,
                                    "bbox": [vx1, vy1, vx2, vy2],
                                }
                            )

                            # Remove this vehicle from the list
                            if best_vehicle in vehicle_detections:
                                vehicle_detections.remove(best_vehicle)

        # Process remaining vehicles without detected plates
        for vehicle in vehicle_detections:
            vx1, vy1, vx2, vy2, v_conf, v_cls = vehicle

            vehicle_type = "truck" if v_cls == 7 else "car"
            draw_color_tuple = (0, 165, 255) if v_cls == 7 else (0, 0, 255)
            status_text = "TRUCK - NO PLATE" if v_cls == 7 else "CAR - NO PLATE"

            cv2.rectangle(display_img, (vx1, vy1), (vx2, vy2), draw_color_tuple, 3)

            label = f"{status_text} ({(v_conf * 100):.1f}%)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                display_img,
                (vx1, vy1 - label_size[1] - 8),
                (vx1 + label_size[0] + 4, vy1),
                draw_color_tuple,
                -1,
            )
            cv2.putText(
                display_img,
                label,
                (vx1 + 2, vy1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            detections.append(
                {
                    "type": vehicle_type,
                    "plate": "no plate",
                    "confidence": v_conf,
                    "color": "orange" if vehicle_type == "truck" else "red",
                    "status": status_text,
                    "bbox": [vx1, vy1, vx2, vy2],
                }
            )

        # Encode the processed image to base64
        _, buffer = cv2.imencode(".jpg", display_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "vehicles": detections,
                "count": len(detections),
                "processed_image": f"data:image/jpeg;base64,{img_base64}",
            }
        )

    except Exception as e:
        print(f"Error in test_image: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"})


@app.route("/add_plate", methods=["POST"])
def add_plate():
    data = request.json
    plate_number = data.get("plate_number", "").strip()
    vehicle_type = data.get("vehicle_type", "truck")

    # Clean plate number: lowercase, no spaces, alphanumeric only
    plate_number = "".join(c.lower() for c in plate_number if c.isalnum())

    if not plate_number:
        return jsonify({"error": "Invalid plate number"})

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO plates (plate_number, vehicle_type) VALUES (?, ?)",
            (plate_number, vehicle_type),
        )
        conn.commit()
        return jsonify({"status": "Plate added successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Plate already exists"})
    finally:
        conn.close()


@app.route("/delete_plate", methods=["POST"])
def delete_plate():
    data = request.json
    plate_number = data.get("plate_number", "").strip()

    # Clean plate number to match stored format
    plate_number = "".join(c.lower() for c in plate_number if c.isalnum())

    if not plate_number:
        return jsonify({"error": "Invalid plate number"})

    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM plates WHERE plate_number = ?", (plate_number,))
        if cursor.rowcount == 0:
            return jsonify({"error": "Plate not found"})
        conn.commit()
        return jsonify({"status": "Plate deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"Delete failed: {str(e)}"})
    finally:
        conn.close()


@app.route("/get_plates")
def get_plates():
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number, vehicle_type FROM plates")
    plates = cursor.fetchall()
    conn.close()

    return jsonify([{"plate_number": p[0], "vehicle_type": p[1]} for p in plates])


def read_license_plate_enhanced(license_plate_crop):
    """Enhanced license plate reading with multiple preprocessing techniques optimized for European plates."""
    if license_plate_crop.size == 0:
        return None, None
    
    # Resize plate to optimal size for OCR - bigger is better for OCR
    target_height = 80  # Increased from 64
    height, width = license_plate_crop.shape[:2]
    if height > 0:
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
        # Make sure width is reasonable - European plates are wider
        target_width = max(200, min(400, target_width))  # Increased minimum width
        plate_resized = cv2.resize(license_plate_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        return None, None
    
    # Convert to grayscale
    if len(plate_resized.shape) == 3:
        gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_resized.copy()
    
    # Try multiple preprocessing approaches optimized for license plates
    preprocessing_methods = []
    
    # Method 1: Simple threshold (often works best for high contrast plates)
    _, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    preprocessing_methods.append(('simple', simple_thresh))
    
    # Method 2: OTSU threshold
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(('otsu', otsu_thresh))
    
    # Method 3: Inverted OTSU (for dark text on light background)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessing_methods.append(('otsu_inv', otsu_inv))
    
    # Method 4: CLAHE + threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    _, clahe_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(('clahe', clahe_thresh))
    
    # Method 5: Gaussian blur + adaptive threshold
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocessing_methods.append(('adaptive', adaptive))
    
    # Method 6: Morphological cleaning
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, morph_thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessing_methods.append(('morph', morph_thresh))
    
    best_result = None
    best_confidence = 0
    
    # Try OCR on each preprocessed image
    for method_name, processed_img in preprocessing_methods:
        try:
            # Save debug image to see what's being processed
            debug_path = f"debug_plates/ocr_{method_name}_frame_{int(time.time())}.jpg"
            os.makedirs("debug_plates", exist_ok=True)
            cv2.imwrite(debug_path, processed_img)
            
            # Configure EasyOCR for European license plates
            detections = ocr_reader.readtext(
                processed_img,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                width_ths=0.3,  # Lower threshold for smaller text
                height_ths=0.3,
                paragraph=False,
                detail=1,
                batch_size=1
            )
            
            for detection in detections:
                bbox, text, score = detection
                
                # Clean the text more aggressively
                text = text.upper().strip()
                text = text.replace(' ', '').replace('-', '').replace('.', '').replace('_', '')
                text = ''.join(c for c in text if c.isalnum())
                
                print(f"    {method_name}: Raw='{detection[1]}' Clean='{text}' Score={score:.3f}")
                
                # More lenient validation for European plates
                if is_valid_european_license_plate(text) and score > 0.1:  # Much lower threshold
                    if score > best_confidence:
                        best_result = text.lower()
                        best_confidence = score
                        print(f"    NEW BEST: '{text}' from {method_name} (conf: {score:.3f})")
                    
        except Exception as e:
            print(f"    OCR error with {method_name}: {e}")
            continue
    
    return best_result, best_confidence

def is_valid_european_license_plate(text):
    """Validate if text looks like a European license plate (more lenient)."""
    if not text or len(text) < 3 or len(text) > 12:  # More lenient length
        return False
    
    # Must have at least one letter OR one number (more lenient)
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)
    
    # Accept if it has either letters or numbers (some plates might be all numbers)
    if not (has_letter or has_number):
        return False
    
    # Common OCR corrections for European plates
    corrected = text
    corrections = {
        'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8', 'D': '0', 'Q': '0'
    }
    
    for wrong, correct in corrections.items():
        corrected = corrected.replace(wrong, correct)
    
    # European plates often have format: XX LC XXX or XX XXX XX etc.
    # Accept any reasonable alphanumeric combination
    return True  # Very lenient for testing

def enhance_plate_crop(plate_crop, padding=10):
    """Add padding and enhance the plate crop before OCR with better quality."""
    if plate_crop.size == 0:
        return plate_crop
    
    # Add more padding around the plate
    height, width = plate_crop.shape[:2]
    
    # Create a larger image with white padding (better for OCR)
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    
    if len(plate_crop.shape) == 3:
        padded = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * 255
        padded[padding:padding+height, padding:padding+width] = plate_crop
    else:
        padded = np.ones((padded_height, padded_width), dtype=np.uint8) * 255
        padded[padding:padding+height, padding:padding+width] = plate_crop
    
    # Apply slight sharpening to the padded image
    if len(padded.shape) == 3:
        gray_padded = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    else:
        gray_padded = padded
        
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray_padded, -1, kernel)
    
    return sharpened

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
