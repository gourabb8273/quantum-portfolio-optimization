from flask import Blueprint, request, jsonify
import os
import pandas as pd

# Blueprint for upload routes
upload_bp = Blueprint('upload_bp', __name__)

# Directory to store the uploaded file
UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'Invalid file format. Only .xlsx files are allowed.'}), 400

    try:
        temp_excel_path = os.path.join(UPLOAD_FOLDER, 'stock_data.xlsx')
        file.save(temp_excel_path)

        df = pd.read_excel(temp_excel_path)

        csv_file_path = os.path.join(UPLOAD_FOLDER, 'stock_data.csv')
        df.to_csv(csv_file_path, index=False)

        os.remove(temp_excel_path)

        return jsonify({'message': 'File uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
