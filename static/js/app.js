let selectedMode = "Sequential";

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
    preview.src = URL.createObjectURL(file);
}

function processVideo() {

    const video = document.getElementById("videoInput").files[0];
    const rows = document.getElementById("rows").value;
    const cols = document.getElementById("cols").value;
    const channel = document.getElementById("channel").value;

    if (!video) {
        alert("Please upload a video");
        return;
    }

    const formData = new FormData();
    formData.append("video", video);
    formData.append("rows", rows);
    formData.append("cols", cols);
    formData.append("channel", channel);
    formData.append("mode", selectedMode);

    document.getElementById("status").innerText = "Process started";

    fetch("/process", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("status").innerText = "Processing completed";
        document.getElementById("preview").src = data.video_url;
    });
}
