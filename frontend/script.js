let mediaRecorder;
let audioChunks = [];

// Switch tabs
function showTab(tab) {
    document.getElementById("voice").style.display = "none";
    document.getElementById("handwriting").style.display = "none";
    document.getElementById(tab).style.display = "block";
}

// 🎤 Start Recording
async function startRecording() {
    audioChunks = [];

    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    mediaRecorder.start();

    document.getElementById("voiceResult").innerText = "Recording...";
}

// 🎤 Stop Recording
function stopRecording() {
    if (!mediaRecorder) {
        document.getElementById("voiceResult").innerText = "Click Start first";
        return;
    }

    // set BEFORE stop
    mediaRecorder.onstop = async () => {
        document.getElementById("voiceResult").innerText = "Processing...";

        let blob = new Blob(audioChunks, { type: 'audio/webm' });

        let formData = new FormData();
        formData.append("file", blob);

        try {
            let response = await fetch("https://parkinsons-detector-lu0v.onrender.com/predict_voice", {
                method: "POST",
                body: formData
            });

            let data = await response.json();

            document.getElementById("voiceResult").innerText =
                data.prediction || data.error;

        } catch (err) {
            document.getElementById("voiceResult").innerText = "Error connecting to server";
        }
    };

    mediaRecorder.stop();
}

// ✍️ Image Upload
async function uploadImage() {
    let fileInput = document.getElementById("imageInput");

    if (!fileInput.files.length) {
        document.getElementById("imageResult").innerText = "Select an image first";
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById("imageResult").innerText = "Processing...";

    try {
        let response = await fetch("https://parkinsons-detector-lu0v.onrender.com/predict_image", {
            method: "POST",
            body: formData
        });

        let data = await response.json();

        document.getElementById("imageResult").innerText =
            data.prediction || data.error;

    } catch (err) {
        document.getElementById("imageResult").innerText = "Error connecting to server";
    }
}