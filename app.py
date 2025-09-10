import os
import requests
from flask import Flask, request, render_template
from PIL import Image
import pytesseract
from gtts import gTTS

app = Flask(__name__)

# Hugging Face Token (vem das variáveis de ambiente no Render)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "Nenhum arquivo enviado", 400

    file = request.files["file"]
    img = Image.open(file.stream)

    # OCR (texto)
    extracted_text = pytesseract.image_to_string(img, lang="eng+por")

    # Classificação de imagem (Hugging Face)
    file.stream.seek(0)  # reseta ponteiro
    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=file.read()
    )

    try:
        predictions = response.json()
        label = predictions[0]["label"]
    except Exception:
        label = "Não identificado"

    # Texto final
    result_text = f"OCR: {extracted_text.strip()} | Classificação: {label}"

    # Converter para áudio
    tts = gTTS(text=result_text, lang="pt")
    audio_path = "static/output.mp3"
    tts.save(audio_path)

    return render_template("index.html", result=result_text, audio=audio_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
