"""
Utility functions for the Traffic Signal Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import cv2
import os
from PIL import Image, ImageEnhance, ImageFilter
import json

class ImageProcessor:
    """Enhanced image processing utilities"""
    
    @staticmethod
    def enhance_image(image_path, output_path=None):
        """Apply image enhancement techniques"""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Apply enhancements
            # 1. Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # 2. Increase sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
            
            # 3. Adjust brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
            
            # 4. Apply slight denoising
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            if output_path:
                img.save(output_path)
                return output_path
            else:
                return img
                
        except Exception as e:
            print(f" Error enhancing image: {e}")
            return None
    
    @staticmethod
    def preprocess_opencv(image_path, target_size=(224, 224)):
        """Advanced preprocessing with OpenCV"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Resize
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f" Error in OpenCV preprocessing: {e}")
            return None
    
    @staticmethod
    def create_image_grid(image_paths, predictions, grid_size=(4, 4), figsize=(15, 15)):
        """Create a grid of images with predictions"""
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
        axes = axes.flatten()
        
        for i, (img_path, pred) in enumerate(zip(image_paths[:len(axes)], predictions[:len(axes)])):
            if i >= len(axes):
                break
                
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{pred['class']}\n{pred['confidence']:.2%}", 
                                fontsize=10, fontweight='bold')
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                           ha='center', va='center')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(image_paths), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

class ModelEvaluator:
    """Model evaluation and analysis utilities"""
    
    @staticmethod
    def evaluate_model_comprehensive(model, test_generator, class_labels):
        """Comprehensive model evaluation"""
        print(" Starting Comprehensive Model Evaluation...")
        
        # Get predictions
        predictions = model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print("=" * 60)
        report = classification_report(y_true, y_pred, 
                                     target_names=class_labels, 
                                     output_dict=True)
        print(classification_report(y_true, y_pred, target_names=class_labels))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        ModelEvaluator.plot_confusion_matrix(cm, class_labels)
        
        # Per-class analysis
        ModelEvaluator.analyze_per_class_performance(report, class_labels)
        
        # Confidence analysis
        ModelEvaluator.analyze_prediction_confidence(predictions, y_true, y_pred, class_labels)
        
        return report
    
    @staticmethod
    def plot_confusion_matrix(cm, class_labels, figsize=(12, 10)):
        """Plot enhanced confusion matrix"""
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Also plot raw counts
        plt.figure(figsize=figsize)
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Oranges',
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def analyze_per_class_performance(report, class_labels):
        """Analyze performance for each class"""
        print("\n Per-Class Performance Analysis:")
        print("=" * 50)
        
        metrics_df = []
        for label in class_labels:
            if label in report:
                precision = report[label]['precision']
                recall = report[label]['recall']
                f1 = report[label]['f1-score']
                support = report[label]['support']
                
                metrics_df.append({
                    'Class': label,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Support': support
                })
                
                # Performance interpretation
                if f1 >= 0.9:
                    performance = "🟢 Excellent"
                elif f1 >= 0.8:
                    performance = "🟡 Good"
                elif f1 >= 0.7:
                    performance = "🟠 Fair"
                else:
                    performance = "🔴 Poor"
                
                print(f"{label.upper():15} | F1: {f1:.3f} | {performance}")
        
        # Plot metrics
        if metrics_df:
            import pandas as pd
            df = pd.DataFrame(metrics_df)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Precision
            axes[0].bar(df['Class'], df['Precision'], color='skyblue', alpha=0.8)
            axes[0].set_title('Precision by Class', fontweight='bold')
            axes[0].set_ylabel('Precision')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_ylim(0, 1)
            
            # Recall
            axes[1].bar(df['Class'], df['Recall'], color='lightgreen', alpha=0.8)
            axes[1].set_title('Recall by Class', fontweight='bold')
            axes[1].set_ylabel('Recall')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_ylim(0, 1)
            
            # F1-Score
            axes[2].bar(df['Class'], df['F1-Score'], color='lightcoral', alpha=0.8)
            axes[2].set_title('F1-Score by Class', fontweight='bold')
            axes[2].set_ylabel('F1-Score')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def analyze_prediction_confidence(predictions, y_true, y_pred, class_labels):
        """Analyze prediction confidence distribution"""
        print("\nPrediction Confidence Analysis:")
        print("=" * 40)
        
        # Get confidence scores
        max_confidences = np.max(predictions, axis=1)
        correct_predictions = (y_true == y_pred)
        
        # Overall confidence stats
        print(f"Average Confidence: {np.mean(max_confidences):.3f}")
        print(f"Confidence (Correct): {np.mean(max_confidences[correct_predictions]):.3f}")
        print(f"Confidence (Incorrect): {np.mean(max_confidences[~correct_predictions]):.3f}")
        
        # Confidence distribution plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Overall confidence distribution
        plt.subplot(2, 2, 1)
        plt.hist(max_confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Overall Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(max_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_confidences):.3f}')
        plt.legend()
        
        # Subplot 2: Confidence by correctness
        plt.subplot(2, 2, 2)
        plt.hist(max_confidences[correct_predictions], bins=15, alpha=0.7, 
                color='green', label='Correct', edgecolor='black')
        plt.hist(max_confidences[~correct_predictions], bins=15, alpha=0.7, 
                color='red', label='Incorrect', edgecolor='black')
        plt.title('Confidence by Prediction Correctness')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Subplot 3: Accuracy vs Confidence
        plt.subplot(2, 2, 3)
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_confidences >= confidence_bins[i]) & (max_confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
        plt.title('Calibration Plot (Accuracy vs Confidence)')
        plt.xlabel('Confidence Score')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Confidence by class
        plt.subplot(2, 2, 4)
        class_confidences = []
        for i, label in enumerate(class_labels):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_conf = np.mean(max_confidences[class_mask])
                class_confidences.append(class_conf)
            else:
                class_confidences.append(0)
        
        plt.bar(range(len(class_labels)), class_confidences, 
               color='purple', alpha=0.7, edgecolor='black')
        plt.title('Average Confidence by Class')
        plt.xlabel('Class')
        plt.ylabel('Average Confidence')
        plt.xticks(range(len(class_labels)), class_labels, rotation=45)
        
        plt.tight_layout()
        plt.show()

class DatasetAnalyzer:
    """Dataset analysis utilities"""
    
    @staticmethod
    def analyze_dataset(dataset_path):
        """Analyze dataset structure and distribution"""
        print(" Dataset Analysis")
        print("=" * 30)
        
        class_counts = {}
        total_images = 0
        
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                class_counts[class_name] = count
                total_images += count
                print(f"{class_name:15} | {count:5d} images")
        
        print(f"{'TOTAL':15} | {total_images:5d} images")
        
        # Visualize distribution
        if class_counts:
            plt.figure(figsize=(12, 6))
            
            # Bar plot
            plt.subplot(1, 2, 1)
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            plt.bar(classes, counts, color='steelblue', alpha=0.8, edgecolor='black')
            plt.title('Class Distribution', fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(counts):
                plt.text(i, v + max(counts) * 0.01, str(v), ha='center', va='bottom')
            
            # Pie chart
            plt.subplot(1, 2, 2)
            plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            plt.title('Class Distribution (Percentage)', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Check for imbalance
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count
            
            print(f"\n Dataset Balance Analysis:")
            print(f"   Imbalance Ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 3:
                print("     Dataset is imbalanced! Consider:")
                print("      • Data augmentation for minority classes")
                print("      • Class weights in training")
                print("      • Oversampling techniques")
            elif imbalance_ratio > 1.5:
                print("    Moderate imbalance detected")
            else:
                print("    Dataset is well balanced")
        
        return class_counts

def save_results(results, filename="prediction_results.json"):
    """Save prediction results to file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f" Results saved to {filename}")
    except Exception as e:
        print(f" Error saving results: {e}")

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f" Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f" Error loading config: {e}")
        return None