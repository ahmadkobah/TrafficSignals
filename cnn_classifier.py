"""
CNN-based Traffic Signal Classifier
Using Transfer Learning (MobileNetV2) for small datasets
"""

import os
import numpy as np
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2


class CNNClassifier:
    """CNN-based traffic signal classifier using Transfer Learning"""
    
    def __init__(self, input_size=(96, 96)):
        self.input_size = input_size
        self.model = None
        self.class_labels = []
        self.history = None
    
    def build_model(self, num_classes):
        """Build model using Transfer Learning (MobileNetV2)"""
        # Use pretrained MobileNetV2 - great for small datasets
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3)
        )
        
        # Freeze base model layers (use pretrained features)
        base_model.trainable = False
        
        # Build our classifier on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_generators(self, train_path, val_path, batch_size=16):
        """Create data generators with augmentation"""
        # Augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            horizontal_flip=False,  # Don't flip traffic signs
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            train_path,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_path,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen
    
    def train(self, train_path, val_path, epochs=30, batch_size=16):
        """Train the CNN model with Transfer Learning"""
        print("\n" + "="*50)
        print("TRAINING CNN CLASSIFIER (Transfer Learning)")
        print("="*50)
        
        # Create generators
        train_gen, val_gen = self.create_generators(train_path, val_path, batch_size)
        
        num_classes = len(train_gen.class_indices)
        self.class_labels = list(train_gen.class_indices.keys())
        
        print(f"\nClasses found: {self.class_labels}")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Using: MobileNetV2 (pretrained on ImageNet)")
        
        # Build model
        print("\nBuilding model...")
        self.model = self.build_model(num_classes)
        
        # Show model summary
        trainable = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable:,}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        print("\nTraining...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation
        val_loss, val_acc = self.model.evaluate(val_gen, verbose=0)
        print(f"\n{'='*50}")
        print(f"[OK] Training complete!")
        print(f"Final Validation Accuracy: {val_acc:.2%}")
        print(f"{'='*50}")
        
        return self.history
    
    def predict(self, image_path, top_k=3):
        """Predict class for an image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.input_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'image_path': image_path,
            'predictions': []
        }
        
        for idx in top_indices:
            results['predictions'].append({
                'class': self.class_labels[idx],
                'confidence': float(predictions[idx]),
                'method': 'CNN (MobileNetV2)'
            })
        
        results['top_class'] = results['predictions'][0]['class']
        results['top_confidence'] = results['predictions'][0]['confidence']
        
        return results
    
    def save_model(self, path='models/cnn_model'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.model.save(f'{path}.h5')
        
        with open(f'{path}_labels.json', 'w') as f:
            json.dump({'classes': self.class_labels}, f, indent=2)
        
        print(f"[OK] CNN model saved to {path}.h5")
    
    def load_model(self, path='models/cnn_model'):
        """Load trained model"""
        self.model = tf.keras.models.load_model(f'{path}.h5')
        
        with open(f'{path}_labels.json', 'r') as f:
            data = json.load(f)
            self.class_labels = data['classes']
        
        print(f"[OK] CNN model loaded from {path}.h5")
        print(f"Classes: {self.class_labels}")


# For testing
if __name__ == "__main__":
    classifier = CNNClassifier()
    classifier.train('dataset/train', 'dataset/validation', epochs=30)
    classifier.save_model('models/cnn_model')
