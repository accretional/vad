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
let decodedSamples = null; // float32 samples at 16kHz after conversion

fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    decodedSamples = null;
    if (selectedFile) {
        uploadText.textContent = selectedFile.name;
        fileInput.parentElement.classList.add('has-file');
        detectBtn.disabled = false;
    }
});

// Decode any audio file to 16kHz mono float32 using Web Audio API
async function decodeToF32(file) {
    const arrayBuffer = await file.arrayBuffer();

    // If it's already raw f32 PCM, use directly
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'f32' || ext === 'pcm' || ext === 'raw') {
        return new Float32Array(arrayBuffer);
    }

    // Decode using AudioContext (handles mp3, wav, ogg, flac, etc.)
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    try {
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        // If already 16kHz, just grab channel 0
        if (audioBuffer.sampleRate === 16000) {
            return audioBuffer.getChannelData(0);
        }

        // Resample to 16kHz using OfflineAudioContext
        const duration = audioBuffer.duration;
        const targetLength = Math.ceil(duration * 16000);
        const offlineCtx = new OfflineAudioContext(1, targetLength, 16000);
        const source = offlineCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineCtx.destination);
        source.start(0);
        const rendered = await offlineCtx.startRendering();
        return rendered.getChannelData(0);
    } finally {
        audioCtx.close();
    }
}

// Create a playable audio blob from a slice of float32 samples at 16kHz
function samplesToWavBlob(samples) {
    const sampleRate = 16000;
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // WAV header
    writeStr(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(view, 8, 'WAVE');
    writeStr(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
    view.setUint16(32, numChannels * bytesPerSample, true);
    view.setUint16(34, bitsPerSample, true);
    writeStr(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Convert float32 [-1,1] to int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeStr(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

detectBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    detectBtn.disabled = true;
    showStatus('Decoding audio to 16kHz mono...', 'info');
    resultsEl.classList.add('hidden');
    try {
        // Decode file to 16kHz float32 if not already done
        if (!decodedSamples) {
            decodedSamples = await decodeToF32(selectedFile);
        }

        showStatus('Processing audio...', 'info');

        // Send raw float32 bytes to server
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: decodedSamples.buffer,
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
            <td></td>
        `;
        // Add play button if we have decoded samples
        if (decodedSamples) {
            const playCell = row.querySelector('td:last-child');
            const btn = document.createElement('button');
            btn.className = 'play-btn';
            btn.textContent = 'Play';
            btn.addEventListener('click', () => playSegment(seg.start, seg.end));
            playCell.appendChild(btn);
        }
        tableBody.appendChild(row);
    });
    resultsEl.classList.remove('hidden');
}

let currentAudio = null;

function playSegment(start, end) {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    if (!decodedSamples) return;

    const startSample = Math.floor(start * 16000);
    const endSample = Math.min(Math.ceil(end * 16000), decodedSamples.length);
    const chunk = decodedSamples.slice(startSample, endSample);
    const blob = samplesToWavBlob(chunk);
    const url = URL.createObjectURL(blob);
    currentAudio = new Audio(url);
    currentAudio.addEventListener('ended', () => URL.revokeObjectURL(url));
    currentAudio.play();
}

// --- Auto-close countdown ---
(function() {
    const timerText = document.getElementById('timer-text');
    const pauseBtn = document.getElementById('timer-pause');
    let remaining = 25;
    let paused = false;

    const interval = setInterval(() => {
        if (paused) return;
        remaining--;
        timerText.textContent = 'Closing in ' + remaining + 's';
        if (remaining <= 0) {
            clearInterval(interval);
            window.close();
            timerText.textContent = 'Window close blocked by browser \u2014 close manually';
        }
    }, 1000);

    pauseBtn.addEventListener('click', () => {
        paused = !paused;
        pauseBtn.textContent = paused ? 'Resume' : 'Pause';
    });
})();
