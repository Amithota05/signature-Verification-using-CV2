from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Function to load and preprocess signature images
def load_and_preprocess_image(image_data):
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

# Compare signatures, generate similarity and image of matches
def compare_signatures(reference_image, input_image):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_image, None)
    kp2, des2 = orb.detectAndCompute(input_image, None)

    if des1 is None or des2 is None:
        return 0, None  # No descriptors found, return 0% similarity and no image

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top 10 matches
    result_image = cv2.drawMatches(reference_image, kp1, input_image, kp2, matches[:10], None, flags=2)

    # Calculate similarity
    total_keypoints = max(len(kp1), len(kp2))
    matching_keypoints = len(matches)
    similarity_percentage = (matching_keypoints / total_keypoints) * 100 if total_keypoints > 0 else 0

    # Convert result image to Base64
    _, buffer = cv2.imencode('.jpg', result_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return similarity_percentage, img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    reference_file = request.files['reference']
    input_file = request.files['input']

    reference_image = load_and_preprocess_image(reference_file.read())
    input_image = load_and_preprocess_image(input_file.read())

    similarity_percentage, result_image = compare_signatures(reference_image, input_image)

    return jsonify({'similarity': similarity_percentage, 'image': result_image})

if __name__ == "__main__":
    app.run(debug=True)