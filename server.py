from flask import Flask, request, jsonify
import base64, cv2, numpy as np
import easyocr

app = Flask(__name__)
reader = easyocr.Reader(['en'])  # OCR-Modell initialisieren

@app.route('/api/license-plate', methods=['POST'])
def recognize_plate():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'no image'}), 400

    # Base64 -> Bild
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # OCR anwenden
    results = reader.readtext(img)
    plate_text = ' '.join([r[1] for r in results])

    print("Detected plate:", plate_text)
    return jsonify({'plate': plate_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
