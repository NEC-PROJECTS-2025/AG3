from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
import cv2

app = Flask(__name__)

# Load the model
model = load_model('model/brain_tumor_model.h5')

def is_brain_mri(img_path):
    """
    Validate if the image appears to be a brain MRI scan
    """
    try:
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        # Check image properties
        height, width = img.shape
        
        # Basic size validation
        if height < 50 or width < 50:
            return False
            
        # Check if image is mostly grayscale
        color_img = cv2.imread(img_path)
        if color_img is None:
            return False
            
        # Convert to HSV and check saturation
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        avg_saturation = np.mean(saturation)
        
        # MRI scans should have low saturation (grayscale-like)
        if avg_saturation > 50:
            return False
            
        # Check for typical MRI characteristics
        
        # 1. Calculate histogram for intensity distribution
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten() / (height * width)  # Normalize
        
        # 2. Check for sharp edges (flowcharts typically have more sharp edges)
        edges = cv2.Canny(img, 100, 200)
        edge_ratio = np.count_nonzero(edges) / (height * width)
        if edge_ratio > 0.1:  # If too many sharp edges, likely a diagram/flowchart
            return False
            
        # 3. Check intensity variance (MRIs typically have smooth transitions)
        intensity_var = np.var(img)
        if intensity_var < 100:  # Too uniform, might be a diagram
            return False
            
        # 4. Check for text-like features (common in flowcharts)
        # Use morphological operations to detect text-like structures
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        text_ratio = np.count_nonzero(dilated) / (height * width)
        if text_ratio > 0.15:  # Too much text-like content
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating image: {str(e)}")
        return False

def predict_image(img_path):
    """
    Make prediction on the image
    """
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize((150, 150))  # Resize to model's expected input
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        
        result = {
            'is_tumor': probability > 0.5,
            'probability': probability,
            'prediction': 'Tumor Detected' if probability > 0.5 else 'No Tumor Detected',
            'confidence': f"{abs(probability - 0.5) * 2 * 100:.2f}%"
        }
        
        return result
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        # Save the uploaded file
        upload_path = os.path.join('static', 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        filename = file.filename
        file_path = os.path.join(upload_path, filename)
        file.save(file_path)
        
        # Validate if it's a brain MRI
        if not is_brain_mri(file_path):
            return render_template('error.html',
                                error_message="The uploaded image does not appear to be a brain MRI scan. Please upload a valid MRI scan image.",
                                filename=filename)
        
        # Make prediction
        result = predict_image(file_path)
        if result is None:
            return render_template('error.html',
                                error_message="Error processing the image. Please try again.",
                                filename=filename)
        
        return render_template('result.html',
                             prediction=result['prediction'],
                             confidence=result['confidence'],
                             filename=filename)
                             
    except Exception as e:
        return render_template('error.html',
                             error_message=f"An error occurred: {str(e)}",
                             filename=None)

if __name__ == '__main__':
    app.run(debug=True)