"""
Traffic Signal Classification System
Main entry point - Simple and clean interface
"""

import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from classifier import TrafficSignalClassifier
from cnn_classifier import CNNClassifier


def print_menu():
    """Display main menu"""
    print("\n" + "="*50)
    print("   TRAFFIC SIGNAL CLASSIFIER")
    print("="*50)
    print("\n1. Train KNN/SVM Classifier (HOG features)")
    print("2. Train CNN Classifier (Deep Learning)")
    print("3. Predict using KNN/SVM")
    print("4. Predict using CNN")
    print("5. Open GUI")
    print("6. Exit")
    print()


def train_knn_svm():
    """Train KNN and SVM classifiers"""
    classifier = TrafficSignalClassifier()
    
    train_path = input("Training data path [dataset/train]: ").strip()
    if not train_path:
        train_path = "dataset/train"
    
    val_path = input("Validation data path [dataset/validation]: ").strip()
    if not val_path:
        val_path = "dataset/validation"
    
    try:
        results = classifier.train(train_path, val_path)
        
        save = input("\nSave model? (y/n): ").strip().lower()
        if save == 'y':
            classifier.save_model('models/traffic_classifier')
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")


def train_cnn():
    """Train CNN classifier"""
    classifier = CNNClassifier(input_size=(64, 64))
    
    train_path = input("Training data path [dataset/train]: ").strip()
    if not train_path:
        train_path = "dataset/train"
    
    val_path = input("Validation data path [dataset/validation]: ").strip()
    if not val_path:
        val_path = "dataset/validation"
    
    epochs = input("Number of epochs [20]: ").strip()
    epochs = int(epochs) if epochs else 20
    
    try:
        classifier.train(train_path, val_path, epochs=epochs)
        
        save = input("\nSave model? (y/n): ").strip().lower()
        if save == 'y':
            classifier.save_model('models/cnn_model')
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")


def predict_knn_svm():
    """Predict using KNN/SVM"""
    classifier = TrafficSignalClassifier()
    
    try:
        classifier.load_model('models/traffic_classifier')
    except:
        print("[ERROR] No trained model found. Train first!")
        return
    
    while True:
        image_path = input("\nImage path (or 'q' to quit): ").strip()
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            continue
        
        try:
            # Get method choice
            print("\nMethod: 1=KNN, 2=SVM, 3=Ensemble")
            method_choice = input("Choose [3]: ").strip()
            method = {'1': 'knn', '2': 'svm', '3': 'ensemble'}.get(method_choice, 'ensemble')
            
            results = classifier.predict(image_path, method=method)
            
            print(f"\n{'='*40}")
            print(f"RESULT: {results['top_class'].upper()}")
            print(f"Confidence: {results['top_confidence']:.1%}")
            print(f"{'='*40}")
            print("\nAll predictions:")
            for pred in results['predictions'][:3]:
                bar = "#" * int(pred['confidence'] * 20)
                print(f"  {pred['class']:15} {bar} {pred['confidence']:.1%}")
                
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")


def predict_cnn():
    """Predict using CNN"""
    classifier = CNNClassifier()
    
    try:
        classifier.load_model('models/cnn_model')
    except:
        print("[ERROR] No trained model found. Train first!")
        return
    
    while True:
        image_path = input("\nImage path (or 'q' to quit): ").strip()
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            continue
        
        try:
            results = classifier.predict(image_path, top_k=3)
            
            print(f"\n{'='*40}")
            print(f"RESULT: {results['top_class'].upper()}")
            print(f"Confidence: {results['top_confidence']:.1%}")
            print(f"{'='*40}")
            print("\nAll predictions:")
            for pred in results['predictions']:
                bar = "#" * int(pred['confidence'] * 20)
                print(f"  {pred['class']:15} {bar} {pred['confidence']:.1%}")
                
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")


def open_gui():
    """Open the GUI interface"""
    try:
        from gui import TrafficSignalGUI
        app = TrafficSignalGUI()
        app.run()
    except ImportError as e:
        print(f"[ERROR] GUI not available: {e}")


def main():
    """Main entry point - directly opens GUI"""
    # Check for command line argument to use terminal mode
    if len(sys.argv) > 1 and sys.argv[1] == '--terminal':
        terminal_mode()
    else:
        # Default: Open GUI directly
        open_gui()


def terminal_mode():
    """Terminal mode for advanced users"""
    while True:
        print_menu()
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            train_knn_svm()
        elif choice == '2':
            train_cnn()
        elif choice == '3':
            predict_knn_svm()
        elif choice == '4':
            predict_cnn()
        elif choice == '5':
            open_gui()
        elif choice == '6':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
