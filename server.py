from flask import Flask, request, jsonify
import base64, cv2, numpy as np
import pytesseract

app = Flask(__name__)

@app.route('/api/license-plate', methods=['POST'])
def recognize_plate():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'no image'}), 400

        # Base64 -> Bild umwandeln
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Vorverarbeitung
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        # OCR mit Tesseract
        text = pytesseract.image_to_string(gray, lang='eng')
        text = text.strip()

        print("Detected plate text:", text)
        return jsonify({'plate': text})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
