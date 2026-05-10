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

function processVideo() {

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

    document.getElementById("status").innerText = "Process started";

    fetch("/process", {
        method: "POST",
        body: formData
    })
    .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || "Processing failed");
        }
        return data;
    })
    .then(data => {
        document.getElementById("status").innerText = "Processing completed";
        const preview = document.getElementById("preview");
        const cacheBustedUrl = `${data.video_url}?t=${Date.now()}`;
        preview.src = cacheBustedUrl;
        preview.load();
    })
    .catch(error => {
        document.getElementById("status").innerText = `Error: ${error.message}`;
    });
}

setMode(selectedMode);
