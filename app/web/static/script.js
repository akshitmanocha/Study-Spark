const form = document.getElementById('study-form');
const loading = document.getElementById('loading');
const summaryContainer = document.getElementById('summary-container');
const summaryDiv = document.getElementById('summary');
const errorContainer = document.getElementById('error-container');
const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const recordingIndicator = document.getElementById('recording-indicator');
const audioPlayback = document.getElementById('audio-playback');
const audioFileInput = document.getElementById('audio-file');
const downloadBtn = document.getElementById('download-btn');
const downloadSection = document.getElementById('download-section');

let mediaRecorder;
let audioChunks = [];
let recordedAudioFile;

recordBtn.addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
        audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        audioPlayback.src = audioUrl;
        audioPlayback.classList.remove('hidden');
        recordedAudioFile = new File([audioBlob], "recording.wav", { type: 'audio/wav' });
        audioChunks = [];
    };

    mediaRecorder.start();
    recordBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    recordingIndicator.classList.remove('hidden');
});

stopBtn.addEventListener('click', () => {
    mediaRecorder.stop();
    recordBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    recordingIndicator.classList.add('hidden');
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    loading.classList.remove('hidden');
    summaryContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');
    downloadSection.classList.add('hidden');

    const formData = new FormData();
    const llm = document.getElementById('llm').value;
    formData.append('llm', llm);

    if (audioFileInput.files.length > 0) {
        formData.append('file', audioFileInput.files[0]);
    } else if (recordedAudioFile) {
        formData.append('file', recordedAudioFile);
    } else {
        alert("Please upload or record an audio file.");
        loading.classList.add('hidden');
        return;
    }

    try {
        const response = await fetch('/study/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const data = await response.json();
        summaryDiv.innerHTML = data.summary.replace(/\n/g, '<br>');

        downloadSection.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        errorContainer.textContent = `An error occurred: ${error.message}`;
        errorContainer.classList.remove('hidden');
    } finally {
        loading.classList.add('hidden');
    }
});

downloadBtn.addEventListener('click', () => {
    const summaryText = summaryDiv.innerHTML.replace(/<br>/g, '\n');
    const blob = new Blob([summaryText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'study_guide.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});
