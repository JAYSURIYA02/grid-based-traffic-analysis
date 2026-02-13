const rowsSlider = document.getElementById("rows");
const colsSlider = document.getElementById("cols");

rowsSlider.oninput = () => {
    document.getElementById("rowsValue").innerText = rowsSlider.value;
}

colsSlider.oninput = () => {
    document.getElementById("colsValue").innerText = colsSlider.value;
}

document.getElementById("videoInput").onchange = function(event) {
    const file = event.target.files[0];
    const preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(file);
}

function processVideo() {

    const video = document.getElementById("videoInput").files[0];
    const rows = rowsSlider.value;
    const cols = colsSlider.value;
    const channel = document.getElementById("channel").value;
    const mode = document.getElementById("mode").value;

    const formData = new FormData();
    formData.append("video", video);
    formData.append("rows", rows);
    formData.append("cols", cols);
    formData.append("channel", channel);
    formData.append("mode", mode);

    document.getElementById("progressBar").style.width = "50%";
    document.getElementById("status").innerText = "Processing started...";

    fetch("/process", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("progressBar").style.width = "100%";
        document.getElementById("status").innerText = "Processing running in background.";
    });
}
