from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Dummy model training function
def train_model(data):
    time.sleep(3)  # Simulate training time
    result = f"✅ Model trained successfully on {len(data)} rows!"
    return result

# Home Route - Upload File
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'traffic_data.csv')
            file.save(file_path)
            return render_template('preview.html', data=preview_data(file_path))
    
    return render_template('upload.html')

# Function to preview first 5 rows
def preview_data(file_path):
    df = pd.read_csv(file_path)
    return df.head().to_html(classes='table table-striped', index=False)

# AJAX Route - Train Model
@app.route('/train', methods=['POST'])
def train():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'traffic_data.csv')
    
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'No file uploaded yet.'})
    
    df = pd.read_csv(file_path)
    training_result = train_model(df)
    
    return jsonify({'status': 'success', 'message': training_result})

if __name__ == '__main__':
    app.run(debug=True)
