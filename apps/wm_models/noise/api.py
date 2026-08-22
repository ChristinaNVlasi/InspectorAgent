"""
Production Flask API for Beko RGS Noise Diagnosis and Vision Detection
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from noise_classifier import NoiseClassifier

# Add vision model path
VISION_MODEL_PATH = Path(__file__).parent.parent.parent / "vision_model" / "ai_vision"
sys.path.insert(0, str(VISION_MODEL_PATH))

# Import vision detection components
try:
    from embeddings.clip_embedder import CLIPEmbedder
    from models.rag_inspector import RAGComponentInspector
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Vision detection not available: {e}")
    VISION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Configuration
UPLOAD_FOLDER = '/tmp/beko_rgs_uploads'
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load noise classifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'noise_classifier_model.pkl')
try:
    classifier = NoiseClassifier()
    classifier.load(MODEL_PATH)
    logger.info("Noise classifier model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load noise classifier: {e}")
    classifier = None

# Load vision detection model
vision_inspector = None
if VISION_AVAILABLE:
    try:
        RAG_DATABASE_PATH = VISION_MODEL_PATH / "data" / "rag_databases.pkl"
        if RAG_DATABASE_PATH.exists():
            embedder = CLIPEmbedder()
            vision_inspector = RAGComponentInspector(embedder)
            vision_inspector.load_databases(str(RAG_DATABASE_PATH))
            logger.info("Vision detection model loaded successfully")
        else:
            logger.warning(f"RAG database not found at {RAG_DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load vision detector: {e}")
        vision_inspector = None


def allowed_file(filename, file_type='audio'):
    """Check if file extension is allowed"""
    if not '.' in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'audio':
        return ext in ALLOWED_AUDIO_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'noise_model_loaded': classifier is not None,
        'vision_model_loaded': vision_inspector is not None,
        'noise_classes': classifier.label_names if classifier else [],
        'vision_components': ['cabinet_panel', 'detergent_dispenser', 'front_wall', 'general_surface'] if vision_inspector else []
    })


@app.route('/diagnose', methods=['POST'])
def diagnose():
    """Main diagnosis endpoint"""
    try:
        # Validate model is loaded
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get optional model_id
        model_id = request.form.get('model_id', 'Unknown')
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"Processing audio file: {filename} for model: {model_id}")
        
        # Perform prediction
        result = classifier.predict(filepath)
        
        # Clean up
        os.remove(filepath)
        
        # Format response
        response = {
            'success': True,
            'model_id': model_id,
            'diagnosis': result['prediction'],
            'confidence': round(result['confidence'] * 100, 2),
            'confidence_scores': {
                k: round(v * 100, 2) for k, v in result['all_scores'].items()
            },
            'recommendation': get_recommendation(result['prediction'])
        }
        
        logger.info(f"Diagnosis completed: {result['prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/detect-damage', methods=['POST'])
def detect_damage():
    """Detect damage in washing machine component images"""
    try:
        # Check if vision model is loaded
        if vision_inspector is None:
            logger.error("Vision model not loaded")
            return jsonify({'error': 'Vision model not available'}), 503
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, file_type='image'):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'
            }), 400
        
        # Save uploaded image temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        file.save(filepath)
        logger.info(f"Image saved: {filepath}")
        
        # Detect component and damage
        result = vision_inspector.detect_component_type(filepath)
        
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'component_type': result['component_type'],
            'confidence': round(result['confidence'] * 100, 2),
            'damage_detected': result.get('has_damage', True),
            'damage_description': result.get('damage_description', ''),
            'recommendation': get_damage_recommendation(result['component_type'], result.get('damage_description', ''))
        }
        
        logger.info(f"Damage detection completed: {result['component_type']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during damage detection: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def get_damage_recommendation(component_type, damage_description):
    """Get recommendation based on detected component damage"""
    recommendations = {
        'cabinet_panel': 'Cabinet panel damage detected',
        'detergent_dispenser': 'Detergent dispenser damage detected',
        'front_wall': 'Front wall damage detected',
        'general_surface': 'Surface damage detected'
    }
    
    return recommendations.get(component_type, 'Component damage detected')


def get_recommendation(diagnosis):
    """Get recommendation based on diagnosis"""
    recommendations = {
        'bearing worn out': 'Bearing assembly wear detected',
        'conterweight loose': 'Counterweight needs tightening',
        'foot adjustment wrong': 'Machine leveling incorrect',
        'motor noise': 'Motor issue detected',
        'shock absorber fault': 'Shock absorber fault detected',
        'springs loose': 'Suspension springs loose',
        'water pump faulty': 'Water pump malfunction'
    }
    return recommendations.get(diagnosis, 'Component issue detected')


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
