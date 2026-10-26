# TrafficSignals

TrafficSignals is a desktop application that recognises traffic signs in photos. It trains and
compares two approaches on a small seven-class image dataset: classic machine learning (HOG and
colour features with KNN and SVM) and deep learning (a MobileNetV2 transfer-learning CNN).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Status](#project-status)
3. [How It Works](#how-it-works)
4. [Dataset](#dataset)
5. [Models](#models)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Getting Started](#getting-started)
9. [Configuration and Parameters](#configuration-and-parameters)
10. [Results](#results)
11. [Known Issues](#known-issues)
12. [License](#license)

---

## Overview

### Purpose

The project answers a simple question: given a picture of a traffic sign, which sign is it?
It is a learning and comparison project. The same dataset is used to train a classic computer
vision pipeline and a neural network, so the two approaches can be compared side by side in one
tool.

### What it recognises

Seven sign classes:

| Class folder   | Meaning                  |
|----------------|--------------------------|
| `30zone`       | Speed limit 30           |
| `40zone`       | Speed limit 40           |
| `50zone`       | Speed limit 50           |
| `80zone`       | Speed limit 80           |
| `Stopsign`     | Stop sign                |
| `crosswalk`    | Pedestrian crossing      |
| `nocarallowed` | No entry for cars        |

### Intended users

- Students and developers who want to learn image classification with a small, readable code base.
- Anyone who wants to compare feature-based classifiers (KNN, SVM) with a CNN on the same data.

### Core capabilities

- Train a KNN and an SVM classifier on HOG and HSV colour-histogram features.
- Train a CNN using MobileNetV2 (pretrained on ImageNet) with data augmentation.
- Predict the class of a single image with KNN, SVM, a KNN+SVM ensemble, or the CNN.
- Show the top predictions with a confidence value for each.
- Save and load trained models (`.pkl` for KNN/SVM, `.h5` for the CNN).
- Two front ends: a Tkinter GUI (default) and an interactive terminal menu.

---

## Project Status

| Area                                  | Status                                                  |
|---------------------------------------|---------------------------------------------------------|
| KNN / SVM pipeline (`classifier.py`)  | Implemented                                             |
| CNN pipeline (`cnn_classifier.py`)    | Implemented                                             |
| Tkinter GUI (`gui.py`)                | Implemented                                             |
| Terminal menu (`main.py --terminal`)  | Implemented                                             |
| Trained models in `models/`           | Committed (KNN, SVM and CNN)                            |
| `config.json`                         | Present, but **not read by any code** (see Known Issues)|
| Colab notebook                        | Early prototype with a different, simple CNN            |
| Automated tests                       | None                                                    |

The project is a working prototype. There is no packaging, no test suite and no CI.

---

## How It Works

```
                 +---------------------------+
                 |  dataset/train/<class>/   |
                 |  dataset/validation/...   |
                 +-------------+-------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
 +-----------------------+            +-------------------------+
 | Classic pipeline      |            | Deep learning pipeline  |
 | resize 64x64          |            | resize 96x96, /255      |
 | HOG (grey) + HSV hist |            | augmentation (train)    |
 +-----------+-----------+            +------------+------------+
             |                                     |
     +-------+-------+                             v
     v               v                 +-------------------------+
 +-------+       +-------+             | MobileNetV2 (frozen)    |
 |  KNN  |       |  SVM  |             | + GAP + Dense + softmax |
 | k = 5 |       |  RBF  |             +------------+------------+
 +---+---+       +---+---+                          |
     |               |                              |
     +-------+-------+                              |
             v                                      v
 +-----------------------+            +-------------------------+
 | models/traffic_       |            | models/cnn_model.h5     |
 | classifier_*.pkl      |            | models/cnn_model_       |
 | + _labels.json        |            | labels.json             |
 +-----------+-----------+            +------------+------------+
             |                                     |
             +------------------+------------------+
                                v
                 +-----------------------------+
                 | GUI / terminal: pick image, |
                 | show class + confidence     |
                 +-----------------------------+
```

### Training flow

1. The user chooses the training and validation folders (defaults: `dataset/train` and
   `dataset/validation`). Each sub-folder name is a class label.
2. **KNN/SVM:** every image is read with OpenCV, resized to 64x64 and turned into one feature
   vector (HOG on the grey image plus a 96-bin HSV colour histogram). Labels are encoded with
   `LabelEncoder`. KNN and SVM are trained on the same vectors, and both are scored on the
   training and validation sets. The models are saved as `models/traffic_classifier_knn.pkl`,
   `models/traffic_classifier_svm.pkl` and `models/traffic_classifier_labels.json`.
3. **CNN:** Keras `ImageDataGenerator` streams images from the folders. Training images are
   augmented. A MobileNetV2 base with frozen weights is topped with a small classifier and
   trained with early stopping and learning-rate reduction. The model is saved as
   `models/cnn_model.h5` with `models/cnn_model_labels.json`.

### Prediction flow

1. The user loads a saved model and selects an image (PNG, JPG, JPEG, BMP; the GUI also lists GIF).
2. The image goes through the same preprocessing as in training.
3. The model returns a probability per class:
   - `knn` or `svm`: probabilities from that model.
   - `ensemble`: the average of KNN and SVM probabilities (used by the GUI).
   - CNN: softmax output; the top 3 classes are returned.
4. Results are sorted by confidence and shown with a text bar (terminal) or in the result panel (GUI).

---

## Dataset

The dataset is stored in the repository under `dataset/`, one folder per class.

| Split        | Images per class | Classes | Total images |
|--------------|------------------|---------|--------------|
| `train`      | 60               | 7       | 420          |
| `validation` | 20               | 7       | 140          |

- File types: 480 PNG and 80 JPG files.
- Size on disk: about 15.5 MB (train) and 5.1 MB (validation).
- The split is a fixed 75 / 25 split by folder. There is no separate test set.
- The repository does not state the source or the licence of the images.

---

## Models

### 1. Feature-based classifiers (`classifier.py`)

**Feature extraction (`FeatureExtractor`)**

| Feature             | Settings                                                                 |
|---------------------|--------------------------------------------------------------------------|
| HOG                 | grey image 64x64, 9 orientations, 8x8 pixels per cell, 2x2 cells per block, `L2-Hys` |
| Colour histogram    | HSV, 32 bins per channel (H: 0-180, S and V: 0-256), each normalised     |
| Combined vector     | HOG + colour histogram concatenated (1,764 + 96 = 1,860 values)          |

**Classifiers (`TrafficSignalClassifier`)**

| Model    | Settings                                                     |
|----------|--------------------------------------------------------------|
| KNN      | `n_neighbors=5`, Euclidean distance                          |
| SVM      | RBF kernel, `C=10`, `gamma='scale'`, `probability=True`      |
| Ensemble | Mean of KNN and SVM class probabilities                      |

### 2. CNN with transfer learning (`cnn_classifier.py`)

```
Input (96 x 96 x 3, scaled to 0..1)
  -> MobileNetV2 (ImageNet weights, include_top=False, frozen)
  -> GlobalAveragePooling2D
  -> Dropout(0.3)
  -> Dense(128, relu)
  -> Dropout(0.3)
  -> Dense(7, softmax)
```

| Setting             | Value                                                         |
|---------------------|---------------------------------------------------------------|
| Optimizer           | Adam, learning rate 0.001                                     |
| Loss                | Categorical cross-entropy                                     |
| Batch size          | 16                                                            |
| Epochs (default)    | 30 in the GUI, 20 in the terminal menu                        |
| Early stopping      | monitor `val_accuracy`, patience 8, restore best weights      |
| ReduceLROnPlateau   | monitor `val_loss`, factor 0.5, patience 4, min LR 1e-6       |
| Augmentation        | rotation 20, shift 0.15, shear 0.1, zoom 0.15, brightness 0.8-1.2, no horizontal flip |

Horizontal flipping is turned off on purpose, because mirrored signs (for example numbers)
would no longer be valid signs.

### 3. Colab prototype (`ColabSectionfiles/TrafficSignals5.ipynb`)

An earlier experiment run in Google Colab. It trains a small CNN from scratch
(three `Conv2D` + `MaxPooling2D` blocks, `Dense(64)`, softmax) on 150x150 images for 15 epochs,
using paths under `/content/`. It is not used by the application.

---

## Technology Stack

| Area              | Technology                                   |
|-------------------|----------------------------------------------|
| Language          | Python 3 (compiled files show Python 3.13 was used) |
| Deep learning     | TensorFlow / Keras (`>=2.10.0`), MobileNetV2 |
| Classic ML        | scikit-learn (`>=1.0.0`): KNN, SVM, LabelEncoder |
| Feature extraction| scikit-image (`>=0.19.0`) HOG                |
| Image processing  | OpenCV (`>=4.5.0`), Pillow (`>=9.0.0`)        |
| Numerics          | NumPy (`>=1.21.0`)                           |
| GUI               | Tkinter / ttk (part of the Python standard library) |
| Model storage     | `pickle` (KNN/SVM), HDF5 `.h5` (CNN), JSON (labels) |
| Prototype         | Jupyter notebook on Google Colab             |

---

## Project Structure

```
TrafficSignals/
|-- main.py                     Entry point: opens the GUI, or the menu with --terminal
|-- gui.py                      Tkinter GUI (Training and Prediction tabs)
|-- classifier.py               HOG + colour features, KNN, SVM, ensemble
|-- cnn_classifier.py           MobileNetV2 transfer-learning CNN
|-- config.json                 Settings file (currently not used by the code)
|-- requirements.txt            Python dependencies
|-- dataset/
|   |-- train/                  60 images for each of the 7 classes
|   |   |-- 30zone/ 40zone/ 50zone/ 80zone/
|   |   `-- Stopsign/ crosswalk/ nocarallowed/
|   `-- validation/             20 images for each of the 7 classes (same folders)
|-- models/
|   |-- cnn_model.h5                    Trained CNN (about 11.4 MB)
|   |-- cnn_model_labels.json           CNN class order
|   |-- traffic_classifier_knn.pkl      Trained KNN (about 6.3 MB)
|   |-- traffic_classifier_svm.pkl      Trained SVM (about 4.8 MB)
|   `-- traffic_classifier_labels.json  KNN/SVM class order
|-- results/                    Training history plots (PNG)
`-- ColabSectionfiles/
    `-- TrafficSignals5.ipynb   Early Colab prototype
```

---

## Getting Started

### Prerequisites

- Python 3 with Tkinter (included in the standard Windows and macOS installers; on Linux
  install `python3-tk`).
- A Python version supported by your TensorFlow release.
- Internet access the first time a CNN is trained, because Keras downloads the MobileNetV2
  ImageNet weights.

### Installation

```bash
git clone https://github.com/ahmadkobah/TrafficSignals.git
cd TrafficSignals
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root. Model and dataset paths in the code are relative
(`models/...`, `dataset/...`).

### Run the GUI (default)

```bash
python main.py
```

or

```bash
python gui.py
```

- **Training tab:** check the dataset paths, set the CNN epochs, then click *Train KNN/SVM* or
  *Train CNN*. The models are saved to `models/` automatically and overwrite the committed ones.
- **Prediction tab:** click *Load KNN/SVM* or *Load CNN*, then *Select Image*, then
  *Predict (KNN/SVM)* (ensemble) or *Predict (CNN)*.

### Run the terminal menu

```bash
python main.py --terminal
```

```
1. Train KNN/SVM Classifier (HOG features)
2. Train CNN Classifier (Deep Learning)
3. Predict using KNN/SVM
4. Predict using CNN
5. Open GUI
6. Exit
```

The menu asks for paths and options; press Enter to accept the defaults. In option 3 you can
choose KNN, SVM or the ensemble. After training you are asked whether to save the model.

### Train the CNN without any front end

```bash
python cnn_classifier.py
```

This trains on `dataset/train` and `dataset/validation` for up to 30 epochs and saves
`models/cnn_model.h5`.

### Use the classes in your own code

```python
from classifier import TrafficSignalClassifier
from cnn_classifier import CNNClassifier

clf = TrafficSignalClassifier()
clf.load_model('models/traffic_classifier')
print(clf.predict('path/to/sign.png', method='svm')['top_class'])

cnn = CNNClassifier()            # default input size 96x96
cnn.load_model('models/cnn_model')
print(cnn.predict('path/to/sign.png', top_k=3))
```

### Use your own dataset

Create one folder per class under a training folder and a validation folder with the same class
names. Any number of classes works; the labels are read from the folder names.

---

## Configuration and Parameters

All active settings are written directly in the source code:

| What                         | Where                                         |
|------------------------------|-----------------------------------------------|
| HOG / colour feature settings| `FeatureExtractor` in `classifier.py`         |
| KNN and SVM settings         | `TrafficSignalClassifier.__init__`            |
| CNN input size               | `CNNClassifier(input_size=...)`, default 96x96|
| CNN layers and optimizer     | `CNNClassifier.build_model`                   |
| Augmentation, batch size     | `CNNClassifier.create_generators` / `train`   |
| Callbacks                    | `CNNClassifier.train`                         |
| Model file names             | `main.py` and `gui.py` (`models/traffic_classifier`, `models/cnn_model`) |

`config.json` lists other values (input 224x224, batch size 32, 30 epochs, a confidence
threshold of 0.7, top-5 predictions, a `logs/` path and stronger augmentation including
horizontal flip). **No code loads this file**, so changing it has no effect. Treat it as a draft
for a future configuration feature.

---

## Results

- The code prints training and validation accuracy for KNN and SVM, and the final validation
  accuracy for the CNN, at the end of each training run. These numbers are not saved in the
  repository.
- `results/` contains three training plots (`resault training 13102025 2225.png`,
  `training_history_20251208_134119.png`, `training_history_20251208_142551.png`). The current
  code does not create these plots; they come from earlier runs.
- The Colab prototype notebook shows a validation accuracy of 33.3 % for its simple CNN. That run
  used a much smaller, earlier dataset with different class names, so it does not describe the
  current models.

No accuracy for the committed models in `models/` is recorded in the repository.

---

## Known Issues

1. **CNN input size differs between training and prediction in the terminal menu.** Menu option
   2 trains with `CNNClassifier(input_size=(64, 64))`, but option 4 and the GUI load the model
   with the default 96x96. A model trained in the terminal menu will likely fail when used for
   prediction. Training from the GUI or from `cnn_classifier.py` uses 96x96 in both steps.
2. **`config.json` is not used** (see above). Its class order also differs from the order in
   the saved label files.
3. **Training plots are not generated** by the current code, although `results/` holds plots
   from earlier versions.
4. **The GUI updates widgets from a background thread.** Tkinter is not thread-safe, so the
   training log can behave unpredictably on some systems. Detailed Keras training output goes
   to the console, not to the GUI log.
5. **Training in the GUI always overwrites** the committed models in `models/` without asking.
6. **Pickle files run code when loaded.** Only load `.pkl` models that you trust.
7. **Small dataset.** 60 training images per class is small; results can vary strongly
   between runs, and there is no separate test set.
8. **`.h5` is a legacy Keras format.** Newer Keras versions recommend the `.keras` format and may
   show warnings.
9. **Bare `except:` clauses** in `main.py` hide the real reason when a model cannot be loaded.

---

## License

No license file is included. Private. All rights reserved.

The source and licence of the dataset images are not stated in the repository.

Author: Ahmad Kobah
