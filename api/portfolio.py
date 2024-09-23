from flask import Flask, request, jsonify, Blueprint
from services.portfolio_optimizer import PortfolioOptimizer  # Import your class
import os
import pandas as pd

# Blueprint for listing routes
portfolio_bp = Blueprint('portfolio_bp', __name__)
@portfolio_bp.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio():
    # Extract parameters from the request
    data = request.json
    min_return=data.get('min_return', 0)
    max_risk=data.get('max_risk', 0)
    num=data.get('num', 0)
    model_type = data.get('model_type', 'CQM'),
    # Create an instance of PortfolioOptimizer
    optimizer = PortfolioOptimizer(
        stocks=data.get('stocks', ['AAPL', 'MSFT', 'AAL', 'WMT']),
        budget=data.get('budget', 1000),
        bin_size=data.get('bin_size'),
        gamma=data.get('gamma'),
        file_path=data.get('file_path', 'data/basic_data.csv'),
        dates=data.get('dates'),
        model_type=data.get('model_type', 'CQM'),
        alpha=data.get('alpha', 0.005),
        baseline=data.get('baseline', '^GSPC'),
        sampler_args=data.get('sampler_args'),
        t_cost=data.get('t_cost', 0.01),
        verbose=data.get('verbose', True),
        min_return=data.get('min_return', 0),
        max_risk=data.get('max_risk', 0),
    )

    # Call the optimize method
    try:
        # result = optimizer.optimize()
        result = optimizer.run(min_return=min_return, max_risk=max_risk, num=num)
        return jsonify({'message': f'Optimization completed using {model_type} optimizer', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
