from flask import Flask, request, jsonify, render_template
from api.upload import upload_bp
from api.list import list_bp
from api.fetch import fetch_bp
from api.portfolio import portfolio_bp

# Initialize Flask app
app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML file

# Register blueprints from different APIs
app.register_blueprint(upload_bp)  # Register upload API
app.register_blueprint(list_bp)    # Register list API
app.register_blueprint(fetch_bp)
app.register_blueprint(portfolio_bp)

# if __name__ == '__main__':
#     app.run(debug=True)
