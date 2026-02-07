let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const sendBtn = document.getElementById("sendBtn");
const audioPlayback = document.getElementById("audioPlayback");
const resultP = document.getElementById("result");

recordBtn.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        audioChunks = [];
        mediaRecorder.start();

        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioPlayback.src = URL.createObjectURL(audioBlob);

            // Arrêter complètement le micro
            stream.getTracks().forEach(t => t.stop());

            sendBtn.disabled = false;

            sendBtn.onclick = async () => {
                const formData = new FormData();
                formData.append("voice", audioBlob, "voice.webm");

                resultP.textContent = "Analyse en cours...";

                try {
                    const res = await fetch("/analyze", {
                        method: "POST",
                        body: formData
                    });

                    const data = await res.json();

                    if (data.error) {
                        resultP.textContent = "❌ Erreur : " + data.error;
                        return;
                    }

                    resultP.textContent =
                        data.prediction === 1 ? "⚠️ Probable Parkinson" : "✅ Voix normale";

                } catch (err) {
                    resultP.textContent = "Erreur de connexion au serveur.";
                }
            };
        };

        // UI
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        sendBtn.disabled = true;
        resultP.textContent = "";

    } catch (err) {
        resultP.textContent = "❌ Impossible d'accéder au microphone.";
        console.error(err);
    }
};

stopBtn.onclick = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    recordBtn.disabled = false;
    stopBtn.disabled = true;
};
