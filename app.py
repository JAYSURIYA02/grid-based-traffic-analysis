# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify
import os
import json
import subprocess
import threading
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from flask import send_file
import io

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
XML_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "xml")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["XML_UPLOAD_FOLDER"] = XML_UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(XML_UPLOAD_FOLDER):
    os.makedirs(XML_UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_config():
    video = request.files.get("video")
    if video is None or video.filename == "":
        return jsonify({"status": "Error", "message": "Video file is required"}), 400

    xml_file = request.files.get("xml_file")
    rows = int(request.form["rows"])
    cols = int(request.form["cols"])
    channel = request.form["channel"]
    mode = request.form["mode"]

    video_filename = os.path.basename(video.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    video.save(video_path)

    xml_path = ""
    if xml_file and xml_file.filename:
        xml_filename = os.path.basename(xml_file.filename)
        xml_path = os.path.join(app.config["XML_UPLOAD_FOLDER"], xml_filename)
        xml_file.save(xml_path)

    config = {}
    if os.path.exists("user_input_data.json"):
        try:
            with open("user_input_data.json", "r") as f:
                config = json.load(f)
        except Exception:
            pass

    config["video"] = os.path.abspath(video_path)
    config["color_channel"] = channel
    config["grids"] = {"rows": rows, "cols": cols}
    config["execution_mode"] = mode
    config["xml_path"] = os.path.abspath(xml_path) if xml_path else ""
    
    # Remove old calibration if it exists when uploading a new config
    if "pixels_per_meter" in config:
        del config["pixels_per_meter"]

    with open("user_input_data.json", "w") as f:
        json.dump(config, f, indent=4)

    return jsonify({"status": "Success", "message": "Configuration saved"})


@app.route("/calibration_frame", methods=["GET"])
def calibration_frame():
    if not os.path.exists("user_input_data.json"):
        return jsonify({"status": "Error", "message": "Configuration not found"}), 404
        
    try:
        with open("user_input_data.json", "r") as f:
            config = json.load(f)
    except Exception:
        return jsonify({"status": "Error", "message": "Invalid configuration"}), 500
        
    video_path = config.get("video")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"status": "Error", "message": "Video not found"}), 404
        
    frame_number = int(request.args.get("frame_number", 0))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"status": "Error", "message": "Cannot open video"}), 500
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or (total_frames > 0 and frame_number >= total_frames):
        cap.release()
        return jsonify({"status": "Error", "message": "Frame out of range"}), 400
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"status": "Error", "message": "Failed to read frame"}), 500
        
    # === Overlay ROI and Grid from configuration ===
    # Lane1 ROI matches seq.py's single ROI and rparallel_he3.py's roi1.
    # Lane2 ROI only exists in rparallel_he3.py (roi2) and is only relevant
    # when the user has selected Parallel mode.
    LANE1_ROI = (47, 202, 441, 141)
    LANE2_ROI = (500, 202, 441, 141)

    grids_config = config.get("grids", {"rows": 6, "cols": 6})
    num_rows, num_cols = int(grids_config.get("rows", 6)), int(grids_config.get("cols", 6))
    mode = config.get("execution_mode", "Sequential")

    def draw_roi_grid(frame, roi_x, roi_y, roi_width, roi_height, num_rows, num_cols):
        grid_width = roi_width // num_cols
        grid_height = roi_height // num_rows

        # Draw outer ROI boundary
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)

        # Draw vertical lines
        for col in range(1, num_cols):
            x = roi_x + col * grid_width
            cv2.line(frame, (x, roi_y), (x, roi_y + roi_height), (0, 0, 255), 1)

        # Draw horizontal lines
        for row in range(1, num_rows):
            y = roi_y + row * grid_height
            cv2.line(frame, (roi_x, y), (roi_x + roi_width, y), (0, 0, 255), 1)

    draw_roi_grid(frame, *LANE1_ROI, num_rows, num_cols)
    if mode == "Parallel":
        draw_roi_grid(frame, *LANE2_ROI, num_rows, num_cols)
    # ================================================================

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return jsonify({"status": "Error", "message": "Failed to encode frame"}), 500
        
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')


@app.route("/calibrate", methods=["POST"])
def calibrate():
    data = request.json
    if not data:
        return jsonify({"status": "Error", "message": "No data provided"}), 400
        
    try:
        x1 = float(data.get("x1"))
        y1 = float(data.get("y1"))
        x2 = float(data.get("x2"))
        y2 = float(data.get("y2"))
        real_world_distance_m = float(data.get("real_world_distance_m"))
    except (TypeError, ValueError):
        return jsonify({"status": "Error", "message": "Invalid numeric coordinates or distance"}), 400
        
    if real_world_distance_m <= 0:
        return jsonify({"status": "Error", "message": "Real world distance must be positive"}), 400
        
    dx = x2 - x1
    dy = y2 - y1
    pixel_distance = float(np.hypot(dx, dy))
    
    if pixel_distance <= 0:
        return jsonify({"status": "Error", "message": "Pixel distance must be greater than zero"}), 400
        
    pixels_per_meter = pixel_distance / real_world_distance_m
    
    if not os.path.exists("user_input_data.json"):
        return jsonify({"status": "Error", "message": "Configuration not found"}), 404
        
    try:
        with open("user_input_data.json", "r") as f:
            config = json.load(f)
    except Exception:
        return jsonify({"status": "Error", "message": "Invalid configuration"}), 500
        
    config["pixels_per_meter"] = round(pixels_per_meter, 3)
    
    with open("user_input_data.json", "w") as f:
        json.dump(config, f, indent=4)
        
    return jsonify({
        "status": "success",
        "pixels_per_meter": round(pixels_per_meter, 3),
        "warning": None
    })


@app.route("/process", methods=["POST"])
def process_video():
    if not os.path.exists("user_input_data.json"):
        return jsonify({"status": "Error", "message": "Configuration not found. Please upload and calibrate first."}), 400
        
    try:
        with open("user_input_data.json", "r") as f:
            config = json.load(f)
    except Exception:
        return jsonify({"status": "Error", "message": "Invalid configuration"}), 500
        
    if "video" not in config or not os.path.exists(config["video"]):
        return jsonify({"status": "Error", "message": "Video not found. Please re-upload."}), 400
        
    ppm = config.get("pixels_per_meter", 0)
    if ppm <= 0:
        return jsonify({"status": "Error", "message": "Invalid or missing calibration. Please calibrate first."}), 400

    # force=True makes Flask parse the body as JSON even if the Content-Type
    # header isn't exactly "application/json" for some reason (proxy,
    # browser extension, etc.) — silent=True still avoids raising if the
    # body is empty/malformed, falling back to the saved config's mode.
    data = request.get_json(silent=True, force=True) or {}
    mode = data.get("mode", config.get("execution_mode", "Sequential"))

    if mode not in ("Sequential", "Parallel"):
        return jsonify({
            "status": "Error",
            "message": f"Invalid execution mode: {mode}"
        }), 400

    # Keep JSON configuration synchronized with the actual mode used
    config["execution_mode"] = mode

    with open("user_input_data.json", "w") as f:
        json.dump(config, f, indent=4)

    if mode == "Parallel":
        cmd = ["python", "rparallel_he3.py"]
        output_file = "test_output1.mp4"
    else:
        cmd = ["python", "seq.py"]
        output_file = "seq_output.mp4"

    # Verifiable in the Flask console — confirms exactly what was requested
    # vs. what is about to be executed for this /process call.
    print(f"[/process] requested_mode(raw)={data.get('mode')!r} "
          f"resolved_mode={mode!r} cmd={cmd}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return jsonify({"status": "Error", "message": f"Failed to start processing: {str(e)}"}), 500

    if result.returncode != 0:
        return jsonify({
            "status": "Error",
            "message": f"{mode} processing failed",
            "details": result.stderr or result.stdout,
            "mode_used": mode
        }), 500

    if not os.path.exists(output_file):
        return jsonify({
            "status": "Error",
            "message": "Processing completed but output video was not generated",
            "details": result.stdout
        }), 500

    # Move output to static folder
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    static_output = os.path.join(static_dir, output_file)
    try:
        os.replace(output_file, static_output)
    except Exception as e:
        return jsonify({
            "status": "Error",
            "message": f"Failed to move output video: {str(e)}"
        }), 500

    print("Returning video:", f"/static/{output_file}")
    return jsonify({
        "status": "Completed",
        "video_url": f"/static/{output_file}",
        "mode_used": mode
    })


if __name__ == "__main__":
    app.run(debug=True)