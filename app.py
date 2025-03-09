from flask import Flask, request, jsonify
from flask_cors import CORS
from learning_strategies_scoring.api_llm_scoring import LLMScoring

app = Flask(__name__)
CORS(app)

# Initialize with CPU
device = "cuda"

# Initialize the scorer with your model
scorer = LLMScoring('readerbench/qwen2_1.5b_scoring_se_ta_su_pa_v3', device=device)

@app.route('/llmscoring/summary', methods=['POST'])
def score_summary():
    args = request.json
    try:
        prediction = scorer.score(args, 'summary')
    except ValueError as e:
        return str(e), 400
    
    response = jsonify(prediction)
    return response, 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
