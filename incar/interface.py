# Standard library imports
import warnings
import os

# Third-party imports for web functionality and ML
import urllib3
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoConfig

# Configure HuggingFace cache directory for model storage
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_HOME'] = '/root/.cache/huggingface'

# Log GPU availability for debugging purposes
if torch.cuda.is_available():
    print("GPU is available. Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Using CPU")

# Import our custom model handler
from incar.model_runner import ActionArranger

# Initialize the ML model - ActionArranger handles device selection internally
action_arranger = ActionArranger()

# Set up template directory path for Flask
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(os.path.dirname(current_dir), 'templates')
print(f"Template directory: {template_dir}")

# Initialize Flask app with custom template directory
app = Flask(__name__, template_folder=template_dir)

# Configure Cross-Origin Resource Sharing (CORS)
# Allow all origins, methods, and headers for development
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure basic security settings for the Flask app
app.config.update(
    SECRET_KEY=os.urandom(24),  # Generate random secret key on startup
    SESSION_COOKIE_HTTPONLY=True,  # Protect against XSS
    DEBUG=True  # Enable debug mode
)

@app.route('/')
def home():
    """
    Route handler for the home page.
    Returns the main interface template.
    """
    return render_template('index.html')

@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    """
    API endpoint for processing text prompts and generating node sequences.
    
    Methods:
        OPTIONS: Handle CORS preflight requests
        POST: Process the prompt and return matching nodes
    
    Returns:
        JSON response containing:
        - status: 'success' or 'error'
        - data: containing prompt, actions, nodes, and descriptions
        - error message if applicable
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return '', 200
        
    # Validate request content type
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415
        
    # Extract and validate prompt from request
    prompt = request.json.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Process the prompt using our ML model
        actions = action_arranger.rearrange_actions(prompt).strip().split()
        print(f"Actions: {actions}")
        
        # Get matching nodes and their descriptions
        nodes, descriptions = action_arranger.find_affine_nodes(actions)
        
        # Structure the successful response
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
        # Log and return any errors that occur during processing
        app.logger.error(f"Error processing prompt: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': 'An error occurred processing your request'
        }), 500

# Run the Flask app if this file is executed directly
if __name__ == "__main__":
    print("\n=== Development Mode: Test Prompts and Their Node Sequences ===\n")
    app.run(
        debug=False,  # Disable debug mode for production
        host='0.0.0.0',  # Allow external connections
        port=5000  # Run on port 5000
    )