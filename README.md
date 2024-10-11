# Signature-Verification-using-CV2
This project focuses on automating the process of signature verification using computer vision techniques with OpenCV, Flask, and Python. It compares two signature images, processes them to extract keypoints, and evaluates their similarity using the ORB (Oriented FAST and Rotated BRIEF) algorithm.

The project uses Flask for the backend to handle image uploads and return the similarity results and an image showing the matching points. It processes the uploaded signature images by converting them into grayscale and applying binary thresholding before extracting the keypoints and descriptors to compare the signatures. The result includes the similarity percentage and an image with matching points, returned in Base64 format.

Features
Upload two signature images for comparison.
Calculate similarity percentage between the reference and input signatures.
Display an image showing the matching keypoints.
Simple web interface built using Flask for uploading images and viewing results.
Technologies Used
Flask: For building the web interface and handling image uploads.
OpenCV: For image processing and signature comparison.
Python: Core programming language for backend processing.
NumPy: For handling array data structures.
PIL (Pillow): For working with image data.
Base64: For encoding result images to be displayed on the web.
