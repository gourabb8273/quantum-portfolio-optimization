from flask import Blueprint, jsonify
import os

# Blueprint for listing routes
list_bp = Blueprint('list_bp', __name__)

UPLOAD_FOLDER = 'data'

@list_bp.route('/files', methods=['GET'])
def list_files():
    try:
        # List files in the upload folder
        files = os.listdir(UPLOAD_FOLDER)
        return jsonify({'files': files}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
