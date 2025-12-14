# 🧠 Brain Tumor MRI Classification Project

A comprehensive deep learning project for classifying brain tumors from MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

## 📋 Overview

This project implements and compares multiple deep learning models for brain tumor classification:
- **Baseline CNN**: Custom convolutional neural network built from scratch
- **VGG19**: Transfer learning using pre-trained VGG19
- **InceptionV3**: Transfer learning using pre-trained InceptionV3
- **ResNet50**: Transfer learning using pre-trained ResNet50

## 🎯 Features

- **Multiple Model Architectures**: Compare performance across different architectures
- **Data Augmentation**: Improve model generalization
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- **Visualization**: Training curves, confusion matrices, and performance comparisons
- **Production Ready**: Inference scripts for deployment
- **Modular Design**: Easy to extend and customize

## 📁 Project Structure

```
brain-tumor-classification/
│
├── Data/                          # Dataset directory
│   ├── Training/                  # Training images
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/                   # Test images
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── outputs/                       # Output directory (created automatically)
│   ├── models/                    # Saved trained models
│   ├── plots/                     # Visualizations and graphs
│   ├── reports/                   # Performance reports
│   └── logs/                      # Training logs
│
├── brain_tumor_classifier.py      # Main training and evaluation script
├── quick_start.py                 # Simplified script for quick experimentation
├── inference.py                   # Prediction script for new images
├── setup_project.py               # Setup and installation script
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
# Navigate to project directory
cd brain-tumor-classification

# Install dependencies
python setup_project.py
```

### 2. Download Dataset

Download the dataset from Kaggle:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Extract the dataset into the `Data/` directory maintaining the folder structure shown above.

### 3. Quick Training (Simple Version)

For a quick test with reduced epochs:

```bash
python quick_start.py
```

This will:
- Train a simple CNN and VGG19 model
- Run for 10 epochs (faster training)
- Generate basic evaluation metrics

### 4. Full Training (Complete Pipeline)

For comprehensive training and evaluation:

```bash
python brain_tumor_classifier.py
```

This will:
- Train all 4 models (Baseline CNN, VGG19, InceptionV3, ResNet50)
- Run for 30 epochs with early stopping
- Generate detailed reports and visualizations
- Save all models and results

## 📊 Dataset Information

The dataset contains MRI images of brain tumors categorized into 4 classes:

| Class | Description | Typical Count |
|-------|-------------|---------------|
| **Glioma** | Most common type of primary brain tumor | ~1,000+ images |
| **Meningioma** | Tumor that forms on membranes covering brain and spinal cord | ~1,000+ images |
| **Pituitary** | Tumors that form in the pituitary gland | ~1,000+ images |
| **No Tumor** | Normal brain MRI scans without tumors | ~500+ images |

## 🤖 Models

### 1. Baseline CNN
- Custom architecture with 4 convolutional blocks
- Batch normalization and dropout for regularization
- Input size: 224×224×3
- Trainable from scratch

### 2. VGG19 Transfer Learning
- Pre-trained on ImageNet
- Frozen convolutional base
- Custom classification head
- Input size: 224×224×3

### 3. InceptionV3 Transfer Learning
- Pre-trained on ImageNet
- More complex architecture with inception modules
- Input size: 299×299×3
- Better for capturing features at multiple scales

### 4. ResNet50 Transfer Learning
- Pre-trained on ImageNet
- Residual connections for better gradient flow
- Input size: 224×224×3
- Often achieves best performance

## 📈 Training Details

### Hyperparameters
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping)
- **Validation Split**: 20% of training data

### Data Augmentation
- Random horizontal flips
- Random rotations (±10%)
- Random zoom (±10%)

### Callbacks
- **ModelCheckpoint**: Save best model based on validation accuracy
- **EarlyStopping**: Stop if validation loss doesn't improve for 10 epochs
- **ReduceLROnPlateau**: Reduce learning rate when loss plateaus
- **TensorBoard**: Log training for visualization
- **CSVLogger**: Save training history to CSV

## 🔮 Making Predictions

### Single Image Prediction

```python
from inference import BrainTumorPredictor

# Initialize predictor
predictor = BrainTumorPredictor(model_path='outputs/models/VGG19_Transfer_best.keras')

# Make prediction
results = predictor.predict_single_image('path/to/mri_image.jpg')

print(f"Predicted: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.2%}")
```

### Batch Prediction

```python
# Predict on multiple images
results = predictor.predict_batch(
    image_folder='path/to/images/',
    output_csv='predictions.csv'
)
```

### Command Line Usage

```bash
# Single image prediction
python inference.py --model outputs/models/VGG19_Transfer_best.keras --image sample_mri.jpg

# Batch prediction
python inference.py --model outputs/models/VGG19_Transfer_best.keras --folder images/ --output results.csv

# Evaluate on test set
python inference.py --model outputs/models/VGG19_Transfer_best.keras --test Data/Testing/

# Run demo
python inference.py --demo
```

## 📊 Expected Performance

Typical performance metrics (may vary based on training):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline CNN | 85-90% | 0.85-0.90 | 0.85-0.90 | 0.85-0.90 |
| VGG19 Transfer | 92-95% | 0.92-0.95 | 0.92-0.95 | 0.92-0.95 |
| InceptionV3 | 93-96% | 0.93-0.96 | 0.93-0.96 | 0.93-0.96 |
| ResNet50 | 94-97% | 0.94-0.97 | 0.94-0.97 | 0.94-0.97 |

## 📁 Output Files

After training, you'll find:

### Models (`outputs/models/`)
- `{ModelName}_best.keras` - Best model weights for each architecture

### Plots (`outputs/plots/`)
- `{ModelName}_training_history.png` - Training/validation curves
- `{ModelName}_confusion_matrix.png` - Confusion matrix heatmap
- `model_comparison.png` - Bar chart comparing all models

### Reports (`outputs/reports/`)
- `model_comparison.csv` - Tabular comparison of metrics
- `classification_report_{timestamp}.txt` - Detailed text report

### Logs (`outputs/logs/`)
- `{ModelName}_training.csv` - Training history in CSV format
- `{ModelName}_tensorboard/` - TensorBoard logs

## 🛠️ Customization

### Adding New Models

1. Create a new build function in `BrainTumorClassifier`:

```python
def build_custom_model(self, name="Custom_Model"):
    model = Sequential([
        # Your architecture here
    ])
    self.models[name] = model
    return model
```

2. Add to the training pipeline in `run_full_pipeline()`:

```python
models_to_train = [
    # ... existing models ...
    ('Custom_Model', self.build_custom_model),
]
```

### Modifying Hyperparameters

Edit the `__init__` method of `BrainTumorClassifier`:

```python
self.batch_size = 64  # Increase batch size
self.epochs = 50      # Train for more epochs
self.learning_rate = 0.00001  # Lower learning rate
```

## 🐛 Troubleshooting

### Common Issues

1. **"Dataset not found"**
   - Ensure dataset is downloaded and extracted to `Data/` folder
   - Check folder structure matches expected format

2. **"Out of Memory" errors**
   - Reduce batch size in configuration
   - Use smaller image size
   - Close other applications

3. **Poor model performance**
   - Ensure dataset is balanced
   - Try more epochs
   - Experiment with data augmentation
   - Check for data leakage

4. **Module import errors**
   - Run `python setup_project.py` to install dependencies
   - Use virtual environment for clean installation

## 📚 Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- OpenCV (optional)

## 🤝 Contributing

Feel free to:
- Add new model architectures
- Improve preprocessing techniques
- Enhance visualization
- Optimize hyperparameters
- Add new evaluation metrics

## 📜 License

This project is for educational purposes. Please ensure you have proper permissions for any medical imaging data used.

## 🔗 References

- Dataset: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- VGG19: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- InceptionV3: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 📧 Support

For questions or issues, please check the troubleshooting section or create an issue in the project repository.

---

**Happy Training! 🚀**
