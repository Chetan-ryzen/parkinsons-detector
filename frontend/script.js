async function predict() {
    const audio = document.getElementById("audio").files[0];
    const image = document.getElementById("image").files[0];

    if (!audio || !image) {
        alert("Upload both files");
        return;
    }

    const formData = new FormData();
    formData.append("audio", audio);
    formData.append("image", image);

    document.getElementById("result").innerText = "Processing...";

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        document.getElementById("result").innerText =
            "Result: " + data.final_result;

    } catch (err) {
        document.getElementById("result").innerText = "Error connecting backend";
    }
}
