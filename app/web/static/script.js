
document.addEventListener('DOMContentLoaded', () => {
    const processBtn = document.getElementById('process-btn');
    const audioFileInput = document.getElementById('audio-file');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const transcriptionOutput = document.getElementById('transcription-output');
    const summaryOutput = document.getElementById('summary-output');

    processBtn.addEventListener('click', async () => {
        const file = audioFileInput.files[0];
        if (!file) {
            alert("Please select an audio file first.");
            return;
        }

        loadingDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/process/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Something went wrong with the processing.');
            }

            const data = await response.json();

            transcriptionOutput.textContent = data.transcription;
            summaryOutput.textContent = data.summary;

            loadingDiv.classList.add('hidden');
            resultsDiv.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            alert(error.message);
            loading_div.classList.add('hidden');
        }
    });
});
