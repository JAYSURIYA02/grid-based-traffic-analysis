let selectedMode = "Sequential";
let selectedVideoObjectUrl = null;

// Highlight selected mode
function setMode(mode) {
    selectedMode = mode;

    document.getElementById("seqBtn").classList.remove("btn-light");
    document.getElementById("parBtn").classList.remove("btn-light");

    document.getElementById("seqBtn").classList.add("btn-outline-light");
    document.getElementById("parBtn").classList.add("btn-outline-light");

    if (mode === "Sequential") {
        document.getElementById("seqBtn").classList.remove("btn-outline-light");
        document.getElementById("seqBtn").classList.add("btn-light");
    } else {
        document.getElementById("parBtn").classList.remove("btn-outline-light");
        document.getElementById("parBtn").classList.add("btn-light");
    }
}

document.getElementById("videoInput").onchange = function(event) {
    const file = event.target.files[0];
    const preview = document.getElementById("preview");

    if (!file) {
        preview.removeAttribute("src");
        preview.load();
        return;
    }

    if (!file.type || !file.type.startsWith("video/")) {
        alert("Please select a valid video file");
        event.target.value = "";
        preview.removeAttribute("src");
        preview.load();
        return;
    }

    if (selectedVideoObjectUrl) {
        URL.revokeObjectURL(selectedVideoObjectUrl);
    }

    selectedVideoObjectUrl = URL.createObjectURL(file);
    preview.src = selectedVideoObjectUrl;
    preview.load();
};

function uploadConfig() {
    const video = document.getElementById("videoInput").files[0];
    const xmlFile = document.getElementById("xmlInput").files[0];
    const rows = document.getElementById("rows").value;
    const cols = document.getElementById("cols").value;
    const channel = document.getElementById("channel").value;

    if (!video) {
        alert("Please upload a video");
        return;
    }

    if (xmlFile && !xmlFile.name.toLowerCase().endsWith(".xml")) {
        alert("Please upload a valid XML file");
        return;
    }

    const formData = new FormData();
    formData.append("video", video);
    if (xmlFile) {
        formData.append("xml_file", xmlFile);
    }
    formData.append("rows", rows);
    formData.append("cols", cols);
    formData.append("channel", channel);
    formData.append("mode", selectedMode);

    const uploadBtn = document.getElementById("uploadBtn");
    uploadBtn.disabled = true;
    uploadBtn.innerText = "Uploading...";
    
    // Reset previous calibration UI state for a fresh upload
    document.getElementById("calibrationSection").style.display = "none";
    document.getElementById("realWorldDist").value = "";

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || "Upload failed");
        }
        return data;
    })
    .then(data => {
        uploadBtn.disabled = false;
        uploadBtn.innerText = "Configuration Saved";
        
        // Show calibration section
        document.getElementById("calibrationSection").style.display = "block";
        
        // Load calibration image
        const calibImg = document.getElementById("calibImg");
        const cacheBustedUrl = `/calibration_frame?frame_number=0&t=${Date.now()}`;
        calibImg.src = cacheBustedUrl;
        
        calibImg.onload = () => {
            initCanvas();
        };
    })
    .catch(error => {
        uploadBtn.disabled = false;
        uploadBtn.innerText = "Upload & Configure";
        alert(`Error: ${error.message}`);
    });
}

// Calibration Canvas Logic
let calibPoints = [];
const markerRadius = 5;

function initCanvas() {
    const calibImg = document.getElementById("calibImg");
    const canvas = document.getElementById("calibCanvas");
    
    // Match canvas size to displayed image size
    canvas.width = calibImg.clientWidth;
    canvas.height = calibImg.clientHeight;
    
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    calibPoints = []; // reset points
    
    // Disable run analysis if recalibrating
    document.getElementById("runAnalysisBtn").disabled = true;
    document.getElementById("calibResult").style.display = "none";
    document.getElementById("calibWarning").style.display = "none";
}

// Redraw canvas on window resize to maintain alignment
window.addEventListener('resize', () => {
    if (document.getElementById("calibrationSection").style.display !== "none") {
        initCanvas();
    }
});

document.getElementById("calibCanvas").addEventListener("mousedown", function(event) {
    if (calibPoints.length >= 2) {
        // Reset if clicking again after 2 points
        initCanvas();
    }
    
    const rect = this.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    calibPoints.push({x: clickX, y: clickY});
    drawCanvas();
});

function drawCanvas() {
    const canvas = document.getElementById("calibCanvas");
    const ctx = canvas.getContext("2d");
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw points
    ctx.fillStyle = "red";
    calibPoints.forEach((p, index) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, markerRadius, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = "yellow";
        ctx.font = "14px Arial";
        ctx.fillText(`P${index+1}`, p.x + 8, p.y - 8);
        ctx.fillStyle = "red";
    });
    
    // Draw line and distance if 2 points
    if (calibPoints.length === 2) {
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(calibPoints[0].x, calibPoints[0].y);
        ctx.lineTo(calibPoints[1].x, calibPoints[1].y);
        ctx.stroke();
        
        const dist = Math.hypot(calibPoints[1].x - calibPoints[0].x, calibPoints[1].y - calibPoints[0].y);
        
        ctx.fillStyle = "cyan";
        ctx.font = "16px Arial";
        const midX = (calibPoints[0].x + calibPoints[1].x) / 2;
        const midY = (calibPoints[0].y + calibPoints[1].y) / 2;
        ctx.fillText(`${dist.toFixed(1)} px (Display)`, midX + 10, midY - 10);
    }
}

function calculateCalibration() {
    if (calibPoints.length < 2) {
        alert("Please select exactly two points on the image.");
        return;
    }
    
    const realDistInput = document.getElementById("realWorldDist").value;
    const realDist = parseFloat(realDistInput);
    
    if (isNaN(realDist) || realDist <= 0) {
        alert("Please enter a valid positive real-world distance.");
        return;
    }
    
    const calibImg = document.getElementById("calibImg");
    
    // Scale coordinates back to native resolution
    const scaleX = calibImg.naturalWidth / calibImg.clientWidth;
    const scaleY = calibImg.naturalHeight / calibImg.clientHeight;
    
    const nativeX1 = calibPoints[0].x * scaleX;
    const nativeY1 = calibPoints[0].y * scaleY;
    const nativeX2 = calibPoints[1].x * scaleX;
    const nativeY2 = calibPoints[1].y * scaleY;
    
    const payload = {
        x1: nativeX1,
        y1: nativeY1,
        x2: nativeX2,
        y2: nativeY2,
        real_world_distance_m: realDist
    };
    
    const calcBtn = document.getElementById("calcCalibBtn");
    calcBtn.disabled = true;
    calcBtn.innerText = "Calculating...";
    
    fetch("/calibrate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || "Calibration failed");
        }
        return data;
    })
    .then(data => {
        calcBtn.disabled = false;
        calcBtn.innerText = "Calculate Calibration";
        
        document.getElementById("ppmValue").innerText = data.pixels_per_meter;
        document.getElementById("calibResult").style.display = "block";
        
        if (data.warning) {
            const warnEl = document.getElementById("calibWarning");
            warnEl.innerText = data.warning;
            warnEl.style.display = "block";
        } else {
            document.getElementById("calibWarning").style.display = "none";
        }
        
        document.getElementById("runAnalysisBtn").disabled = false;
    })
    .catch(error => {
        calcBtn.disabled = false;
        calcBtn.innerText = "Calculate Calibration";
        alert(`Error: ${error.message}`);
    });
}

function runAnalysis() {
    const runBtn = document.getElementById("runAnalysisBtn");
    runBtn.disabled = true;
    runBtn.innerText = "Processing Video... (This may take a while)";

    console.log(`[runAnalysis] sending mode=${selectedMode}`);

    fetch("/process", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            mode: selectedMode
        })
    })
    .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || "Processing failed");
        }
        return data;
    })
    .then(data => {
        runBtn.disabled = false;
        runBtn.innerText = "Run Analysis";

        console.log(`[runAnalysis] requested mode=${selectedMode} -> server used mode_used=${data.mode_used}, video_url=${data.video_url}`);
        if (data.mode_used && data.mode_used !== selectedMode) {
            console.warn(`[runAnalysis] Mismatch! UI requested "${selectedMode}" but server used "${data.mode_used}".`);
        }

        const preview = document.getElementById("preview");
        // Fully detach the old source first. Just swapping the query string
        // on .src is not always enough to force certain browsers to drop a
        // previously buffered/cached video, which can make it look like the
        // output never changed even though a new file was returned.
        preview.pause();
        preview.removeAttribute("src");
        preview.load();

        const cacheBustedUrl = `${data.video_url}?t=${Date.now()}`;
        preview.src = cacheBustedUrl;
        preview.load();
        
        // Scroll to video
        preview.scrollIntoView({ behavior: 'smooth', block: 'center' });
    })
    .catch(error => {
        runBtn.disabled = false;
        runBtn.innerText = "Run Analysis";
        alert(`Error: ${error.message}`);
    });
}

setMode(selectedMode);
