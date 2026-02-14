from flask import Flask, render_template, request, jsonify
import os
import json
import subprocess
import threading

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_video():

    video = request.files["video"]
    rows = int(request.form["rows"])
    cols = int(request.form["cols"])
    channel = request.form["channel"]
    mode = request.form["mode"]

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    config = {
        "video": video_path,
        "color_channel": channel,
        "grids": {"rows": rows, "cols": cols},
        "execution_mode": mode
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
