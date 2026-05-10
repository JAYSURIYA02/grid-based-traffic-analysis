from flask import Flask, render_template, request, jsonify
import os
import json
import subprocess
import threading

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

@app.route("/process", methods=["POST"])
def process_video():

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

    config = {
        "video": os.path.abspath(video_path),
        "color_channel": channel,
        "grids": {"rows": rows, "cols": cols},
        "execution_mode": mode,
        "xml_path": os.path.abspath(xml_path) if xml_path else ""
    }

    with open("user_input_data.json", "w") as f:
        json.dump(config, f, indent=4)

    # Run synchronously (wait until finished)
    if mode == "Parallel":
        subprocess.run(["python", "rparallel_he3.py"])
        output_file = "test_output1.mp4"
    else:
        subprocess.run(["python", "seq.py"])
        output_file = "seq_output.mp4"

    # Move output to static folder
    static_output = os.path.join("static", output_file)
    os.replace(output_file, static_output)
    print("Returning video:", f"/static/{output_file}")
    return jsonify({
        "status": "Completed",
        "video_url": f"/static/{output_file}"
    })


if __name__ == "__main__":
    app.run(debug=True)
