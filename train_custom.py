"""
Example: Advanced Training Script for Traffic Signal Classifier
Customize this script for your specific needs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TrafficSignalClassifier

def train_custom_model():
    """Train a model with custom parameters"""
    
    # Create classifier instance
    classifier = TrafficSignalClassifier()
    
    # Customize training parameters
    TRAIN_PATH = r"C:\path\to\your\dataset\train"          # Change this path
    VALIDATION_PATH = r"C:\path\to\your\dataset\validation"  # Change this path
    EPOCHS = 30                    # Number of training epochs
    MODEL_SAVE_PATH = "my_traffic_model.h5"  # Where to save the model
    
    print("Starting Custom Training...")
    print(f"Training Data: {TRAIN_PATH}")
    print(f"Validation Data: {VALIDATION_PATH}")
    print(f"Epochs: {EPOCHS}")
    
    try:
        # Train the model
        history = classifier.train_model(
            train_path=TRAIN_PATH,
            validation_path=VALIDATION_PATH,
            epochs=EPOCHS,
            model_save_path=MODEL_SAVE_PATH
        )
        
        print("Training completed successfully!")
        
        # Show training plots
        classifier.plot_training_history()
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    train_custom_model()