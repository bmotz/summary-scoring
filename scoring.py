from flask import Flask, request, jsonify
from flask_cors import CORS
from learning_strategies_scoring.api_llm_scoring import LLMScoring
import datetime

app = Flask(__name__)
CORS(app)

# Startup info
startup_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@app.route('/startup-info')
def startup_info():
    return f"App started at: {startup_time}"

# Global variable to hold the scorer (lazy loaded)
scorer = None

def get_scorer():
    """Lazy load the scorer only when first needed"""
    global scorer
    if scorer is None:
        scorer = LLMScoring('upb-nlp/llama32_3b_scoring_all_tasks')
    return scorer

@app.route('/', methods=['OPTIONS'])
@app.route('/score/<task>', methods=['OPTIONS'])
def handle_options(task=None):
    """Handle preflight CORS requests."""
    return '', 204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }

@app.route('/', methods=['POST'])
def score_summary():
    args = request.json
    
    # Check if the request has the old format (with 'question' field)
    if 'question' in args:
        # Remove 'question' field
        args_filtered = {k: v for k, v in args.items() if k != 'question'}
    else:
        args_filtered = args
    
    try:
        # Get scorer only when needed
        scorer = get_scorer()
        prediction = scorer.score(args_filtered, 'summary')
    except ValueError as e:
        return str(e), 400
    
    response = jsonify(prediction)
    return response, 200

@app.route('/score/<task>', methods=['POST'])
def score_task(task):
    if task not in ["selfexplanation", "thinkaloud", "summary", "paraphrasing"]:
        return "Invalid Task (should be one of: 'selfexplanation', 'thinkaloud', 'summary', 'paraphrasing')", 400
    
    args = request.json
    try:
        # Get scorer only when needed
        scorer = get_scorer()
        prediction = scorer.score(args, task)
    except ValueError as e:
        return str(e), 400
    
    response = jsonify(prediction)
    return response, 200

@app.route('/confidence', methods=['OPTIONS'])
def handle_confidence_options():
    """Handle preflight CORS requests for confidence endpoint."""
    return '', 204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }

@app.route('/confidence', methods=['POST'])
def score_summary_with_confidence():
    """Score summary task with confidence metrics"""
    args = request.json
    
    # Check if the request has the old format (with 'question' field)
    if 'question' in args:
        # Remove 'question' field
        args_filtered = {k: v for k, v in args.items() if k != 'question'}
    else:
        args_filtered = args
    
    try:
        # Get scorer only when needed
        scorer = get_scorer()
        result = scorer.score_with_confidence(args_filtered, 'summary')
    except ValueError as e:
        return str(e), 400
    except Exception as e:
        # Handle cases where confidence scoring might fail
        return f"Confidence scoring error: {str(e)}", 500
    
    response = jsonify(result)
    return response, 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
