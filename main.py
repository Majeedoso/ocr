import os
import cv2
import easyocr
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import threading

# Initialize Flask app
app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image to improve OCR accuracy
def preprocess_image(file_stream):
    # Read the image file from the file stream
    img_array = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return None

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding for better contrast
    _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

    return thresh_img

# OCR endpoint
@app.route('/ocr', methods=['POST'])
def ocr():
    # Check if file is present in the request
    if 'file' not in request.files or not request.files['file']:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only jpg, jpeg, and png allowed.'}), 400

    # Preprocess image to improve OCR accuracy
    img = preprocess_image(file)
    if img is None:
        return jsonify({'error': 'Failed to read image. Please upload a valid image.'}), 400

    # Initialize EasyOCR Reader for both Arabic and English
    try:
        reader = easyocr.Reader(['ar', 'en'], gpu=False)  # Arabic and English OCR
    except Exception as e:
        return jsonify({'error': f'Failed to initialize OCR reader: {str(e)}'}), 500

    # Perform OCR with EasyOCR
    try:
        text_results = reader.readtext(img)
    except Exception as e:
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500
    
    # Extract all lines
    all_lines = [t[1] for t in text_results]

    # Return all lines as a single list
    return jsonify({
        'all_lines': all_lines
    })

# Function to run the Flask server
def run_server():
    port = int(os.environ.get('PORT', 6789))
    # Disable Flask’s debug mode to make it run as a background service
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)

# Run the Flask server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.start()

# Optional: You can add other tasks here that will run in the main thread
print("Server is running in the background. You can perform other tasks here.")