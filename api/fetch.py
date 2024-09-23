from flask import Blueprint, jsonify
import os
import pandas as pd

# Blueprint for listing routes
fetch_bp = Blueprint('fetch_bp', __name__)

UPLOAD_FOLDER = 'data'

# @list_bp.route('/files', methods=['GET'])
# def list_files():
#     try:
#         # List files in the upload folder
#         files = os.listdir(UPLOAD_FOLDER)
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# API to fetch first 5 rows of stock data
@fetch_bp.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    try:
        csv_file_path = os.path.join(UPLOAD_FOLDER, 'stock_data.csv')
        if not os.path.exists(csv_file_path):
            return jsonify({'error': 'No stock data available'}), 404

        # Read the CSV file and get first 5 rows
        df = pd.read_csv(csv_file_path)
        data = df.head(5).values.tolist()  # Get first 5 rows as a list
        columns = df.columns.tolist()  # Get the column headers

        return jsonify({'columns': columns, 'data': data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500