"""
Traffic Signal Classifier using KNN, SVM with HOG Features
- HOG (Histogram of Oriented Gradients) for feature extraction
- KNN (K-Nearest Neighbors) classifier
- SVM (Support Vector Machine) classifier
"""

import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
import pickle
import json


class FeatureExtractor:
    """Extract features using HOG and color histograms"""
    
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
    
    def extract_hog(self, image):
        """Extract HOG features from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, self.img_size)
        
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return features
    
    def extract_color_histogram(self, image):
        """Extract color histogram features"""
        image = cv2.resize(image, self.img_size)
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize and flatten
        hist = np.concatenate([
            cv2.normalize(hist_h, hist_h).flatten(),
            cv2.normalize(hist_s, hist_s).flatten(),
            cv2.normalize(hist_v, hist_v).flatten()
        ])
        return hist
    
    def extract_all_features(self, image):
        """Combine HOG and color histogram features"""
        hog_features = self.extract_hog(image)
        color_features = self.extract_color_histogram(image)
        return np.concatenate([hog_features, color_features])


class TrafficSignalClassifier:
    """Main classifier using KNN and SVM with HOG features"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
        self.label_encoder = LabelEncoder()
        self.class_labels = []
        self.is_trained = False
    
    def load_dataset(self, dataset_path):
        """Load images and labels from dataset folder"""
        features = []
        labels = []
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        classes = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not classes:
            raise ValueError("No class folders found in dataset")
        
        print(f"Found {len(classes)} classes: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                image = cv2.imread(img_path)
                
                if image is not None:
                    feature = self.feature_extractor.extract_all_features(image)
                    features.append(feature)
                    labels.append(class_name)
        
        return np.array(features), np.array(labels)
    
    def train(self, train_path, validation_path=None):
        """Train KNN and SVM classifiers"""
        print("\n" + "="*50)
        print("TRAINING TRAFFIC SIGNAL CLASSIFIER")
        print("="*50)
        
        # Load training data
        print("\nLoading training data...")
        X_train, y_train = self.load_dataset(train_path)
        
        if len(X_train) < 5:
            raise ValueError("Not enough training samples. Need at least 5 images.")
        
        # Encode labels
        self.label_encoder.fit(y_train)
        self.class_labels = list(self.label_encoder.classes_)
        y_encoded = self.label_encoder.transform(y_train)
        
        print(f"\nTotal training samples: {len(X_train)}")
        print(f"Feature vector size: {X_train.shape[1]}")
        
        # Train KNN
        print("\nTraining KNN classifier...")
        self.knn.fit(X_train, y_encoded)
        knn_train_acc = self.knn.score(X_train, y_encoded)
        print(f"  KNN Training Accuracy: {knn_train_acc:.2%}")
        
        # Train SVM
        print("\nTraining SVM classifier...")
        self.svm.fit(X_train, y_encoded)
        svm_train_acc = self.svm.score(X_train, y_encoded)
        print(f"  SVM Training Accuracy: {svm_train_acc:.2%}")
        
        # Validate if validation path provided
        if validation_path and os.path.exists(validation_path):
            print("\nEvaluating on validation set...")
            X_val, y_val = self.load_dataset(validation_path)
            y_val_encoded = self.label_encoder.transform(y_val)
            
            knn_val_acc = self.knn.score(X_val, y_val_encoded)
            svm_val_acc = self.svm.score(X_val, y_val_encoded)
            
            print(f"  KNN Validation Accuracy: {knn_val_acc:.2%}")
            print(f"  SVM Validation Accuracy: {svm_val_acc:.2%}")
        
        self.is_trained = True
        print("\n[OK] Training completed!")
        
        return {
            'knn_train_acc': knn_train_acc,
            'svm_train_acc': svm_train_acc,
            'classes': self.class_labels
        }
    
    def predict(self, image_path, method='ensemble'):
        """
        Predict traffic signal class
        method: 'knn', 'svm', or 'ensemble'
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Load and extract features
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        features = self.feature_extractor.extract_all_features(image)
        features = features.reshape(1, -1)
        
        results = {'image_path': image_path, 'predictions': []}
        
        if method == 'knn' or method == 'ensemble':
            knn_proba = self.knn.predict_proba(features)[0]
            knn_pred = np.argmax(knn_proba)
            knn_conf = knn_proba[knn_pred]
            
            if method == 'knn':
                for i, prob in enumerate(knn_proba):
                    results['predictions'].append({
                        'class': self.class_labels[i],
                        'confidence': float(prob),
                        'method': 'KNN'
                    })
        
        if method == 'svm' or method == 'ensemble':
            svm_proba = self.svm.predict_proba(features)[0]
            svm_pred = np.argmax(svm_proba)
            svm_conf = svm_proba[svm_pred]
            
            if method == 'svm':
                for i, prob in enumerate(svm_proba):
                    results['predictions'].append({
                        'class': self.class_labels[i],
                        'confidence': float(prob),
                        'method': 'SVM'
                    })
        
        if method == 'ensemble':
            # Average probabilities from both classifiers
            ensemble_proba = (knn_proba + svm_proba) / 2
            
            for i, prob in enumerate(ensemble_proba):
                results['predictions'].append({
                    'class': self.class_labels[i],
                    'confidence': float(prob),
                    'method': 'Ensemble'
                })
        
        # Sort by confidence
        results['predictions'].sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add top prediction summary
        top = results['predictions'][0]
        results['top_class'] = top['class']
        results['top_confidence'] = top['confidence']
        
        return results
    
    def save_model(self, path='models/classifier'):
        """Save trained models"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save KNN
        with open(f'{path}_knn.pkl', 'wb') as f:
            pickle.dump(self.knn, f)
        
        # Save SVM
        with open(f'{path}_svm.pkl', 'wb') as f:
            pickle.dump(self.svm, f)
        
        # Save label encoder and class labels
        with open(f'{path}_labels.json', 'w') as f:
            json.dump({
                'classes': self.class_labels,
                'encoder_classes': list(self.label_encoder.classes_)
            }, f, indent=2)
        
        print(f"[OK] Models saved to {path}_*.pkl")
    
    def load_model(self, path='models/classifier'):
        """Load trained models"""
        # Load KNN
        with open(f'{path}_knn.pkl', 'rb') as f:
            self.knn = pickle.load(f)
        
        # Load SVM
        with open(f'{path}_svm.pkl', 'rb') as f:
            self.svm = pickle.load(f)
        
        # Load labels
        with open(f'{path}_labels.json', 'r') as f:
            data = json.load(f)
            self.class_labels = data['classes']
            self.label_encoder.classes_ = np.array(data['encoder_classes'])
        
        self.is_trained = True
        print(f"[OK] Models loaded from {path}")
        print(f"Classes: {self.class_labels}")
