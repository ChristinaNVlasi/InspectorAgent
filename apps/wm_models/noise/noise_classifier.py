"""
Similarity-based Noise Classifier for Beko RGS
Uses reference embeddings and cosine similarity for classification
"""
import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoiseClassifier:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.reference_embeddings = {}
        self.label_names = []
    
    def build_reference_embeddings(self, data_dir):
        """Build reference embeddings from training data with augmentation"""
        logger.info(f"Building reference embeddings from {data_dir}")
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.m4a')]
        
        for fname in files:
            label = fname.replace('.m4a', '')
            file_path = os.path.join(data_dir, fname)
            
            logger.info(f"Processing {fname}...")
            
            # Load audio
            y = self.audio_processor.load_audio(file_path)
            
            # Generate augmented versions
            augmented_audios = self.audio_processor.augment_audio(y, num_augmentations=10)
            
            # Extract features for each augmented version
            embeddings = []
            for aug_audio in augmented_audios:
                features = self.audio_processor.extract_features(aug_audio)
                embeddings.append(features)
            
            # Store all embeddings for this class
            self.reference_embeddings[label] = np.array(embeddings)
            self.label_names.append(label)
            
            logger.info(f"Created {len(embeddings)} reference embeddings for '{label}'")
        
        logger.info(f"Total classes: {len(self.label_names)}")
        return self
    
    def predict(self, audio_path):
        """Predict noise class using similarity matching"""
        try:
            # Load and process input audio
            y = self.audio_processor.load_audio(audio_path)
            input_features = self.audio_processor.extract_features(y).reshape(1, -1)
            
            # Calculate similarity with all reference embeddings
            similarities = {}
            for label, embeddings in self.reference_embeddings.items():
                # Calculate cosine similarity with all reference embeddings of this class
                sims = cosine_similarity(input_features, embeddings)
                # Use maximum similarity as the score for this class
                similarities[label] = np.max(sims)
            
            # Get prediction and confidence
            predicted_label = max(similarities, key=similarities.get)
            confidence = similarities[predicted_label]
            
            logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
            
            return {
                'prediction': predicted_label,
                'confidence': float(confidence),
                'all_scores': {k: float(v) for k, v in similarities.items()}
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save(self, filepath='noise_classifier_model.pkl'):
        """Save the classifier"""
        model_data = {
            'reference_embeddings': self.reference_embeddings,
            'label_names': self.label_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath='noise_classifier_model.pkl'):
        """Load the classifier"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.reference_embeddings = model_data['reference_embeddings']
        self.label_names = model_data['label_names']
        logger.info(f"Model loaded from {filepath}")
        return self


if __name__ == '__main__':
    # Train and save the model
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    classifier = NoiseClassifier()
    classifier.build_reference_embeddings(data_dir)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'noise_classifier_model.pkl')
    classifier.save(model_path)
    
    print("\n" + "="*60)
    print("Model training completed successfully!")
    print(f"Classes: {', '.join(classifier.label_names)}")
    print(f"Model saved to: {model_path}")
    print("="*60)
