async function uploadFile() {
    let file = document.getElementById('fileUpload').files[0];
    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch('/predict', {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        console.error("Error:", await response.text());
        return alert("Error processing the file.");
    }

    let result = await response.json();
    let dataset = result.data;

    if (!dataset || dataset.length === 0) {
        return alert("Error: No data received.");
    }

    let anomalies = dataset.filter(d => d.Anomaly === -1).length;
    let normal = dataset.length - anomalies;

    document.getElementById("accuracy-score").textContent = "Accuracy: " + result.accuracy + "%";

    new Chart(document.getElementById('chart'), {
        type: 'pie',
        data: {
            labels: ["Normal", "Anomalies"],
            datasets: [{
                data: [normal, anomalies],
                backgroundColor: ["green", "red"]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

// ✅ Ensure download functions are defined
function downloadCSV() {
    window.location.href = "/download_csv";
}

function downloadReport() {
    window.location.href = "/generate_report";
}
