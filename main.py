"""
Enhanced Traffic Signal Classification System
Features:
- Improved CNN architecture with batch normalization and dropout
- Confidence scores and prediction probabilities
- GUI interface for easy image testing
- Model optimization with callbacks
- Data visualization and performance metrics
- Robust image preprocessing
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

class TrafficSignalClassifier:
    def __init__(self):
        self.model = None
        self.class_labels = []
        self.history = None
        self.input_size = (224, 224)  # Increased for better accuracy
        
    def create_enhanced_model(self, num_classes):
        """Create an improved CNN architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.input_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def create_data_generators(self, train_path, validation_path):
        """Create enhanced data generators with better augmentation"""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2  # Use some training data for validation if needed
        )
        
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.input_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_path,
            target_size=self.input_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def get_callbacks(self, model_path):
        """Create training callbacks for optimization"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train_model(self, train_path, validation_path, epochs=30, model_save_path='traffic_signal_model.h5'):
        """Train the enhanced model"""
        print("Starting Enhanced Traffic Signal Training...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(train_path, validation_path)
        
        # Get number of classes and labels
        num_classes = len(train_gen.class_indices)
        self.class_labels = list(train_gen.class_indices.keys())
        
        print(f"Detected {num_classes} classes: {self.class_labels}")
        
        # Create enhanced model
        self.model = self.create_enhanced_model(num_classes)
        
        print("Model Architecture:")
        self.model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks(model_save_path)
        
        # Train the model
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            validation_data=val_gen,
            validation_steps=len(val_gen),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final model
        val_loss, val_acc, val_top_k = self.model.evaluate(val_gen, verbose=0)
        print(f"Final Validation Results:")
        print(f"   Loss: {val_loss:.4f}")
        print(f"   Accuracy: {val_acc:.4f}")
        print(f"   Top-K Accuracy: {val_top_k:.4f}")
        
        # Save class labels
        with open('class_labels.json', 'w') as f:
            json.dump(self.class_labels, f)
        
        return self.history
    
    def load_model(self, model_path, labels_path='class_labels.json'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.class_labels = json.load(f)
        else:
            print("Warning: Class labels file not found!")
        
        print(f"Model loaded successfully!")
        print(f"📋 Classes: {self.class_labels}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Load and resize image
            img = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=self.input_size
            )
            
            # Convert to array and normalize
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            return img_array, img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None, None
    
    def predict_with_confidence(self, image_path, top_k=3):
        """Make prediction with confidence scores"""
        if self.model is None:
            print("No model loaded!")
            return None
        
        img_array, original_img = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'predictions': [],
            'image_path': image_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for i, idx in enumerate(top_indices):
            confidence = float(predictions[idx])
            class_name = self.class_labels[idx] if idx < len(self.class_labels) else f"Class_{idx}"
            
            results['predictions'].append({
                'rank': i + 1,
                'class': class_name,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-K Accuracy
        if 'top_k_categorical_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], label='Training Top-K')
            axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], label='Validation Top-K')
            axes[1, 0].set_title('Top-K Categorical Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

class TrafficSignalGUI:
    def __init__(self):
        self.classifier = TrafficSignalClassifier()
        self.root = tk.Tk()
        self.root.title("Traffic Signal Classifier")
        self.root.geometry("800x600")
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Traffic Signal Classifier", 
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="5")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model).grid(row=0, column=0, padx=5)
        ttk.Button(model_frame, text="Train New Model", 
                  command=self.train_model).grid(row=0, column=1, padx=5)
        
        # Image selection
        image_frame = ttk.LabelFrame(main_frame, text="Image Classification", padding="5")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(image_frame, text="Select Image", 
                  command=self.select_image).grid(row=0, column=0, padx=5)
        ttk.Button(image_frame, text="Predict", 
                  command=self.predict_image).grid(row=0, column=1, padx=5)
        
        # Image display
        self.image_label = ttk.Label(main_frame, text="No image selected")
        self.image_label.grid(row=3, column=0, pady=10)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=15, width=60)
        self.results_text.grid(row=3, column=1, columnspan=2, pady=10, padx=10)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=3, column=3, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.selected_image_path = None
        
    def load_model(self):
        """Load a trained model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                self.classifier.load_model(model_path)
                self.status_var.set(f"Model loaded: {os.path.basename(model_path)}")
                self.log_message("Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def train_model(self):
        """Train a new model (simplified interface)"""
        train_path = filedialog.askdirectory(title="Select Training Dataset Folder")
        if not train_path:
            return
            
        val_path = filedialog.askdirectory(title="Select Validation Dataset Folder")
        if not val_path:
            return
        
        # Run training in separate thread to avoid GUI freeze
        def train_thread():
            try:
                self.status_var.set("Training in progress...")
                self.log_message("Starting model training...")
                
                history = self.classifier.train_model(train_path, val_path, epochs=20)
                
                self.status_var.set("Training completed!")
                self.log_message("Training completed successfully!")
                
                # Show training plots
                self.classifier.plot_training_history()
                
            except Exception as e:
                messagebox.showerror("Training Error", f"Training failed: {e}")
                self.status_var.set("Training failed")
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def select_image(self):
        """Select an image for classification"""
        file_path = filedialog.askopenfilename(
            title="Select Traffic Signal Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.display_image(file_path)
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        """Display selected image in the GUI"""
        try:
            # Load and resize image for display
            img = Image.open(image_path)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.log_message(f"Error displaying image: {e}")
    
    def predict_image(self):
        """Predict the selected image"""
        if not self.selected_image_path:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if self.classifier.model is None:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        try:
            self.status_var.set("Predicting...")
            
            # Make prediction
            results = self.classifier.predict_with_confidence(self.selected_image_path, top_k=5)
            
            if results:
                self.display_results(results)
                self.status_var.set("Prediction completed")
            else:
                self.log_message("Prediction failed")
                self.status_var.set("Prediction failed")
                
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed: {e}")
            self.status_var.set("Prediction failed")
    
    def display_results(self, results):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        output = f"PREDICTION RESULTS\n"
        output += f"{'='*50}\n"
        output += f"Image: {os.path.basename(results['image_path'])}\n"
        output += f"Time: {results['timestamp']}\n\n"
        
        output += f"TOP PREDICTIONS:\n"
        output += f"{'-'*30}\n"
        
        for pred in results['predictions']:
            confidence_bar = "█" * int(pred['percentage'] / 5)  # Scale to 20 chars max
            output += f"{pred['rank']}. {pred['class'].upper()}\n"
            output += f"   Confidence: {pred['percentage']:.2f}% {confidence_bar}\n"
            output += f"   Score: {pred['confidence']:.4f}\n\n"
        
        # Add interpretation
        best_pred = results['predictions'][0]
        if best_pred['percentage'] > 80:
            confidence_level = "Very High"
        elif best_pred['percentage'] > 60:
            confidence_level = "High"
        elif best_pred['percentage'] > 40:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        output += f"ANALYSIS:\n"
        output += f"{'-'*20}\n"
        output += f"Predicted Signal: {best_pred['class'].upper()}\n"
        output += f"Confidence Level: {confidence_level}\n"
        output += f"Recommendation: "
        
        if best_pred['percentage'] > 70:
            output += "High confidence prediction\n"
        elif best_pred['percentage'] > 50:
            output += "Moderate confidence - verify manually\n"
        else:
            output += "Low confidence - manual verification required\n"
        
        self.results_text.insert(tk.END, output)
        self.log_message(f"Predicted: {best_pred['class']} ({best_pred['percentage']:.1f}% confidence)")
    
    def log_message(self, message):
        """Add message to results area"""
        current_text = self.results_text.get(1.0, tk.END)
        if current_text.strip():
            self.results_text.insert(tk.END, f"\n{message}")
        else:
            self.results_text.insert(tk.END, message)
        
        self.results_text.see(tk.END)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function to choose between training or GUI"""
    print("Enhanced Traffic Signal Classifier")
    print("=" * 40)
    print("Choose an option:")
    print("1. Train new model (command line)")
    print("2. Open GUI interface")
    print("3. Quick test with existing model")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Command line training
        classifier = TrafficSignalClassifier()
        
        train_path = input("Enter training dataset path: ").strip()
        val_path = input("Enter validation dataset path: ").strip()
        epochs = int(input("Enter number of epochs (default 20): ") or "20")
        
        history = classifier.train_model(train_path, val_path, epochs)
        classifier.plot_training_history()
        
    elif choice == "2":
        # GUI interface
        app = TrafficSignalGUI()
        app.run()
        
    elif choice == "3":
        # Quick test
        classifier = TrafficSignalClassifier()
        
        model_path = input("Enter model path (or press Enter for default): ").strip()
        if not model_path:
            model_path = "traffic_signal_model.h5"
        
        try:
            classifier.load_model(model_path)
            
            image_path = input("Enter image path to test: ").strip()
            results = classifier.predict_with_confidence(image_path, top_k=3)
            
            if results:
                print(f"\n🎯 Prediction Results:")
                for pred in results['predictions']:
                    print(f"{pred['rank']}. {pred['class']}: {pred['percentage']:.2f}%")
            
        except Exception as e:
            print(f"Error: {e}")
        
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()