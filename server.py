# ======================================================
# SmartPark AI Light Server
# Kennzeichen-Erkennung mit EasyOCR (CPU-kompatibel)
# Autor: Elias (SmartParkAI)
# ======================================================

from flask import Flask, request, jsonify
import base64, cv2, numpy as np, os, tempfile
from easyocr import Reader

# Initialisiere EasyOCR nur einmal (schneller bei mehreren Aufrufen)
reader = Reader(['en'], gpu=False)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ SmartPark AI Light Server is running"

@app.route('/api/license-plate', methods=['POST'])
def recognize_plate():
    try:
        # Empfange JSON-Daten
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'no image'}), 400

        # Base64 → OpenCV-Bild
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'invalid image'}), 400

        # Graustufen + Filterung
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Temporäre Datei für OCR
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, gray)

        # OCR ausführen
        results = reader.readtext(temp_path)
        os.remove(temp_path)

        # Text extrahieren (nur größtes Feld)
        text = ""
        if results:
            text = max(results, key=lambda x: x[1])[1]
            text = text.strip()

        print("✅ Detected plate:", text)
        return jsonify({'plate': text})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 SmartPark AI Light Server is starting...")
    app.run(host='0.0.0.0', port=5000)
