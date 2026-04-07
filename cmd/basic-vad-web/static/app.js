const fileInput = document.getElementById('audio-file');
const detectBtn = document.getElementById('detect-btn');
const uploadText = document.getElementById('upload-text');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const summarySegments = document.getElementById('summary-segments');
const summarySpeakers = document.getElementById('summary-speakers');
const summaryDuration = document.getElementById('summary-duration');
const tableBody = document.querySelector('#segments-table tbody');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        uploadText.textContent = selectedFile.name;
        fileInput.parentElement.classList.add('has-file');
        detectBtn.disabled = false;
    }
});

detectBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    detectBtn.disabled = true;
    showStatus('Processing audio...', 'info');
    resultsEl.classList.add('hidden');

    try {
        const arrayBuffer = await selectedFile.arrayBuffer();
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: arrayBuffer,
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(text || response.statusText);
        }

        const data = await response.json();
        displayResults(data);
        showStatus(`Detected ${data.segments.length} segments`, 'success');
    } catch (err) {
        showStatus('Error: ' + err.message, 'error');
    } finally {
        detectBtn.disabled = false;
    }
});

function showStatus(msg, type) {
    statusEl.textContent = msg;
    statusEl.className = 'status ' + type;
    statusEl.classList.remove('hidden');
}

function displayResults(data) {
    const segments = data.segments || [];
    const speakers = new Set(segments.map(s => s.speaker_id));

    summarySegments.textContent = `${segments.length} segments`;
    summarySpeakers.textContent = `${speakers.size} speakers`;
    summaryDuration.textContent = `${data.duration.toFixed(2)}s duration`;

    tableBody.innerHTML = '';
    segments.forEach((seg, i) => {
        const dur = (seg.end - seg.start).toFixed(3);
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${seg.start.toFixed(3)}</td>
            <td>${seg.end.toFixed(3)}</td>
            <td>${dur}</td>
            <td class="speaker-${seg.speaker_id}">Speaker ${seg.speaker_id}</td>
            <td>${(seg.confidence * 100).toFixed(1)}%</td>
        `;
        tableBody.appendChild(row);
    });

    resultsEl.classList.remove('hidden');
}
