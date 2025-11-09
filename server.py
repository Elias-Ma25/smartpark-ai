# ======================================================
# SmartPark AI Light Server (Render-Optimierte Version)
# Kennzeichen-Erkennung mit EasyOCR (CPU-kompatibel)
# Autor: Elias (SmartParkAI)
# ======================================================

from flask import Flask, request, jsonify
import base64, cv2, numpy as np, os, tempfile
from easyocr import Reader

# Initialisiere EasyOCR nur einmal
reader = Reader(['en'], gpu=False)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ SmartPark AI Light Server is running (Render Optimized)"

@app.route('/api/license-plate', methods=['POST'])
def recognize_plate():
    try:
        # 1️⃣ JSON prüfen
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image field in request'}), 400

        img_b64 = data['image']
        print(f"📥 Received Base64 length: {len(img_b64)}")

        # 2️⃣ Base64 in Bild umwandeln
        try:
            img_data = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("❌ Base64 decode error:", e)
            return jsonify({'error': f'Invalid base64 data: {str(e)}'}), 400

        if img is None:
            print("❌ cv2.imdecode returned None (invalid image data)")
            return jsonify({'error': 'Invalid image content'}), 400

        # 3️⃣ Bild vorverarbeiten (Graustufen + Filter)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # 4️⃣ Temporäre Datei für OCR
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, gray)

        # 5️⃣ OCR ausführen
        results = []
        try:
            results = reader.readtext(temp_path)
        except Exception as e:
            print("❌ OCR-Error:", e)
            os.remove(temp_path)
            return jsonify({'error': f'OCR failed: {str(e)}'}), 500

        os.remove(temp_path)

        # 6️⃣ Text extrahieren
        text = ""
        if results:
            # größtes Textfeld mit hoher Wahrscheinlichkeit
            text = max(results, key=lambda x: x[2])[1].strip()

        print(f"✅ Detected plate: {text if text else 'None'}")

        return jsonify({'plate': text or "unbekannt"}), 200

    except Exception as e:
        print("❌ Global error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 SmartPark AI Light Server is starting (Render Mode)...")
    app.run(host='0.0.0.0', port=5000)
