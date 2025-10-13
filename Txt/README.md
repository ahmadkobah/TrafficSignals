# 🚦 Enhanced Traffic Signal Classifier

An advanced computer vision system for classifying traffic signals and road signs using deep learning. This enhanced version includes improved accuracy, user-friendly interface, and comprehensive analysis tools.

## 🌟 Key Features

### 🏗️ **Improved Model Architecture**
- **Enhanced CNN** with batch normalization and dropout
- **Larger input size** (224x224) for better feature extraction
- **Global Average Pooling** to reduce overfitting
- **Optimized layer configuration** for traffic sign recognition

### 🎯 **Advanced Prediction System**
- **Confidence scoring** with percentage display
- **Top-K predictions** to see alternative classifications
- **Confidence threshold analysis** for reliability assessment
- **Real-time prediction** with detailed results

### 🖥️ **User-Friendly GUI Interface**
- **Easy-to-use graphical interface** for non-technical users
- **Drag-and-drop image selection**
- **Real-time image preview**
- **Comprehensive results display**
- **Model management** (load/train/save)

### ⚡ **Model Optimization**
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** for optimal convergence
- **Best model checkpointing** to save optimal weights
- **Advanced data augmentation** for robust training

### 📊 **Comprehensive Analytics**
- **Training visualization** with loss/accuracy plots
- **Confusion matrix analysis** with heatmaps
- **Per-class performance metrics** (precision, recall, F1)
- **Confidence distribution analysis**
- **Dataset balance analysis**

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ahmadkobah/TrafficSignals.git
cd TrafficSignals

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

Choose from three options:
1. **Train new model** (command line)
2. **Open GUI interface** (recommended for most users)
3. **Quick test** with existing model

### 3. For GUI Usage

1. Click "**Load Model**" to load a pre-trained model
2. Click "**Select Image**" to choose a traffic signal image
3. Click "**Predict**" to get classification results
4. View detailed results with confidence scores

## 📁 Project Structure

```
TrafficSignals/
├── main.py              # Enhanced main application
├── utils.py             # Utility functions and analysis tools
├── config.json          # Configuration settings
├── requirements.txt     # Python dependencies
├── test_enhanced.py     # Test suite for all features
├── TrafficSignals5.ipynb # Original Jupyter notebook
├── models/              # Saved model files
├── logs/                # Training logs
└── results/             # Prediction results
```

## 🎛️ Configuration

Edit `config.json` to customize:

```json
{
  "model_settings": {
    "input_size": [224, 224],
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.001
  },
  "data_augmentation": {
    "rotation_range": 30,
    "zoom_range": 0.3,
    "brightness_range": [0.8, 1.2]
  },
  "prediction": {
    "confidence_threshold": 0.7,
    "top_k_predictions": 5
  }
}
```

## 📊 Dataset Requirements

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── stop/           # Stop sign images
│   ├── yield/          # Yield sign images
│   ├── speed_limit/    # Speed limit sign images
│   ├── warning/        # Warning sign images
│   └── ...
└── validation/
    ├── stop/
    ├── yield/
    ├── speed_limit/
    ├── warning/
    └── ...
```

## 🔧 Advanced Usage

### Command Line Training

```python
from main import TrafficSignalClassifier

# Create classifier
classifier = TrafficSignalClassifier()

# Train model
history = classifier.train_model(
    train_path="dataset/train",
    validation_path="dataset/validation",
    epochs=30
)

# Visualize training
classifier.plot_training_history()
```

### Batch Prediction

```python
# Load model
classifier.load_model("traffic_signal_model.h5")

# Predict multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
for img_path in images:
    results = classifier.predict_with_confidence(img_path, top_k=3)
    print(f"{img_path}: {results['predictions'][0]['class']} "
          f"({results['predictions'][0]['percentage']:.1f}%)")
```

### Dataset Analysis

```python
from utils import DatasetAnalyzer

# Analyze dataset balance
analyzer = DatasetAnalyzer()
class_counts = analyzer.analyze_dataset("dataset/train")
```

## 📈 Model Performance

### Enhanced Architecture Benefits

| Feature | Original | Enhanced | Improvement |
|---------|----------|----------|-------------|
| Input Size | 150×150 | 224×224 | +48% resolution |
| Layers | 6 layers | 12+ layers | Better feature extraction |
| Regularization | None | Batch norm + Dropout | Reduced overfitting |
| Callbacks | None | 3 callbacks | Optimized training |
| Data Augmentation | Basic | Advanced | +6 techniques |

### Typical Performance Metrics

- **Accuracy**: 85-95% (depending on dataset)
- **Training Time**: 20-40 minutes (GPU recommended)
- **Inference Speed**: ~50ms per image
- **Model Size**: ~15-25 MB

## 🛠️ Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install tensorflow opencv-python pillow matplotlib
   ```

2. **GPU not detected**
   ```bash
   pip install tensorflow-gpu  # For older TF versions
   # Or check CUDA installation
   ```

3. **Low accuracy**
   - Check dataset quality and balance
   - Increase training epochs
   - Adjust learning rate in config.json

4. **Out of memory errors**
   - Reduce batch size in config.json
   - Use smaller input size
   - Close other applications

### Performance Tips

1. **For better accuracy:**
   - Use high-quality, diverse training images
   - Balance your dataset across classes
   - Increase input resolution if you have enough data
   - Use data augmentation appropriately

2. **For faster training:**
   - Use GPU acceleration
   - Reduce input size for initial experiments
   - Use transfer learning with pre-trained models

3. **For deployment:**
   - Convert to TensorFlow Lite for mobile
   - Use model quantization for smaller size
   - Implement batch prediction for multiple images

## 📚 API Reference

### TrafficSignalClassifier Class

#### Methods

- `create_enhanced_model(num_classes)`: Create improved CNN architecture
- `train_model(train_path, validation_path, epochs)`: Train the model
- `load_model(model_path, labels_path)`: Load pre-trained model
- `predict_with_confidence(image_path, top_k)`: Get predictions with confidence
- `plot_training_history()`: Visualize training metrics

#### Example Output

```python
{
  'predictions': [
    {'rank': 1, 'class': 'stop', 'confidence': 0.89, 'percentage': 89.0},
    {'rank': 2, 'class': 'yield', 'confidence': 0.07, 'percentage': 7.0},
    {'rank': 3, 'class': 'speed_limit', 'confidence': 0.04, 'percentage': 4.0}
  ],
  'image_path': 'test_image.jpg',
  'timestamp': '2025-10-13 14:30:22'
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- OpenCV community for image processing tools
- Traffic sign datasets contributors
- Open source community for inspiration and tools

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/ahmadkobah/TrafficSignals/issues)
- **Discussions**: [Community discussions](https://github.com/ahmadkobah/TrafficSignals/discussions)
- **Email**: [Your contact email]

---

**Made with ❤️ for traffic safety and computer vision education**