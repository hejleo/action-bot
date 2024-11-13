import warnings
import urllib3
from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS
import torch
from transformers import AutoConfig

# Set HuggingFace cache directory
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_HOME'] = '/root/.cache/huggingface'

# Check GPU availability for logging
if torch.cuda.is_available():
    print("GPU is available. Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Using CPU")

# Import model runner
from incar.model_runner import ActionArranger

# Initialize model (ActionArranger handles device selection internally)
action_arranger = ActionArranger()

# Get the directory of the current file and set up template path
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(os.path.dirname(current_dir), 'templates')
print(f"Template directory: {template_dir}")

# Create the Flask app with the custom template folder
app = Flask(__name__, template_folder=template_dir)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Basic security configurations
app.config.update(
    SECRET_KEY=os.urandom(24),
    SESSION_COOKIE_HTTPONLY=True,
    DEBUG=True
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 200
        
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415
        
    prompt = request.json.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Extract actions from the prompt and split into list
        actions = action_arranger.rearrange_actions(prompt).strip().split()
        print(f"Actions: {actions}")
        # Get both nodes and their descriptions
        nodes, descriptions = action_arranger.find_affine_nodes(actions)
        
        # Structure the response
        response = {
            'status': 'success',
            'data': {
                'prompt': prompt,
                'actions': actions,
                'nodes': nodes,
                'descriptions': descriptions
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"Error processing prompt: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': 'An error occurred processing your request'
        }), 500

if __name__ == "__main__":
    print("\n=== Development Mode: Test Prompts and Their Node Sequences ===\n")
    app.run(
        debug=False,
        host='0.0.0.0',  # Allow external connections
        port=5000
    )