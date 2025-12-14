"""
Brain Tumor MRI Classification Project
======================================
A comprehensive implementation for classifying brain tumors using deep learning.
Compares multiple models: Baseline CNN, VGG19, InceptionV3, and ResNet50.

Classes:
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor

Author: Claude AI Assistant
Date: November 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Flatten, 
    Dropout, BatchNormalization, GlobalAveragePooling2D,
    Rescaling, RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

# Evaluation imports
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class BrainTumorClassifier:
    """Main class for brain tumor classification project"""
    
    def __init__(self, data_dir='Data', output_dir='outputs'):
        """
        Initialize the classifier
        
        Args:
            data_dir: Path to the dataset directory
            output_dir: Path to save outputs (models, plots, reports)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.data_dir / 'Training'
        self.test_dir = self.data_dir / 'Testing'
        
        # Create output directories
        self.models_dir = self.output_dir / 'models'
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'
        self.logs_dir = self.output_dir / 'logs'
        
        for dir_path in [self.models_dir, self.plots_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.img_size_default = 224
        self.img_size_inception = 299
        self.batch_size = 32
        self.epochs = 30
        self.learning_rate = 0.0001
        
        # Data variables
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.num_classes = 4
        
        # Models dictionary
        self.models = {}
        self.histories = {}
        self.evaluations = {}
        
        print("Brain Tumor Classifier initialized!")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def check_dataset(self):
        """Check if dataset exists and print information"""
        if not self.train_dir.exists() or not self.test_dir.exists():
            print("\n⚠️ Dataset not found!")
            print("Please download the dataset from:")
            print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            print(f"\nExpected structure:")
            print(f"{self.data_dir}/")
            print(f"├── Training/")
            print(f"│   ├── glioma/")
            print(f"│   ├── meningioma/")
            print(f"│   ├── notumor/")
            print(f"│   └── pituitary/")
            print(f"└── Testing/")
            print(f"    ├── glioma/")
            print(f"    ├── meningioma/")
            print(f"    ├── notumor/")
            print(f"    └── pituitary/")
            return False
        
        # Count images in each class
        print("\n📊 Dataset Information:")
        print("-" * 50)
        
        for split in ['Training', 'Testing']:
            split_dir = self.data_dir / split
            print(f"\n{split} Set:")
            total = 0
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                    print(f"  {class_dir.name:12} : {count:5} images")
                    total += count
            print(f"  {'Total':12} : {total:5} images")
        
        return True
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ], name="data_augmentation")
        return data_augmentation
    
    def load_data(self, img_size=224, augment=True):
        """
        Load and preprocess the data
        
        Args:
            img_size: Size to resize images to
            augment: Whether to apply data augmentation
        """
        print(f"\n📥 Loading data with image size {img_size}x{img_size}...")
        
        # Load training data and split into train/validation
        full_train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            image_size=(img_size, img_size),
            batch_size=self.batch_size,
            label_mode='categorical',
            seed=42
        )
        
        # Get class names
        self.class_names = full_train_dataset.class_names
        print(f"Classes: {self.class_names}")
        
        # Split training data into train (80%) and validation (20%)
        train_size = int(0.8 * len(full_train_dataset))
        self.train_dataset = full_train_dataset.take(train_size)
        self.val_dataset = full_train_dataset.skip(train_size)
        
        # Load test data
        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            image_size=(img_size, img_size),
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_dataset = self.train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_dataset = self.val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = self.test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        print("✅ Data loaded successfully!")
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    '''
    def build_baseline_cnn(self, name="Baseline_CNN"):
        """Build a simple CNN from scratch"""
        print(f"\n🏗️ Building {name}...")
        
        model = Sequential([
            # Input and normalization
            Input(shape=(self.img_size_default, self.img_size_default, 3)),
            Rescaling(1./255),
            
            # Data augmentation
            self.create_data_augmentation(),
            
            # Conv Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Conv Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Conv Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Conv Block 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ], name=name)
        
        self.models[name] = model
        print(f"✅ {name} built successfully!")
        return model
    
    def build_vgg19_model(self, name="VGG19_Transfer"):
        """Build VGG19 transfer learning model"""
        print(f"\n🏗️ Building {name}...")
        
        # Load pre-trained VGG19
        base_model = VGG19(
            input_shape=(self.img_size_default, self.img_size_default, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model
        inputs = Input(shape=(self.img_size_default, self.img_size_default, 3))
        x = self.create_data_augmentation()(inputs)
        x = tf.keras.layers.Lambda(lambda x: vgg_preprocess(x))(x)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=name)
        
        self.models[name] = model
        print(f"✅ {name} built successfully!")
        return model 
    ''' 

    
    def build_inception_v3_model(self, name="InceptionV3_Transfer"):
        """Build InceptionV3 transfer learning model"""
        print(f"\n🏗️ Building {name}...")
        
        # Need to reload data with 299x299 for Inception
        self.load_data(img_size=self.img_size_inception)
        
        # Load pre-trained InceptionV3
        base_model = InceptionV3(
            input_shape=(self.img_size_inception, self.img_size_inception, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model
        inputs = Input(shape=(self.img_size_inception, self.img_size_inception, 3))
        x = self.create_data_augmentation()(inputs)
        x = tf.keras.layers.Lambda(lambda x: inception_preprocess(x))(x)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=name)
        
        self.models[name] = model
        print(f"✅ {name} built successfully!")
        return model
    
    def build_resnet50_model(self, name="ResNet50_Transfer"):
        """Build ResNet50 transfer learning model"""
        print(f"\n🏗️ Building {name}...")
        
        # Reload data with 224x224 if needed
        if hasattr(self, '_last_img_size') and self._last_img_size != self.img_size_default:
            self.load_data(img_size=self.img_size_default)
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            input_shape=(self.img_size_default, self.img_size_default, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model
        inputs = Input(shape=(self.img_size_default, self.img_size_default, 3))
        x = self.create_data_augmentation()(inputs)
        x = tf.keras.layers.Lambda(lambda x: resnet_preprocess(x))(x)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=name)
        
        self.models[name] = model
        print(f"✅ {name} built successfully!")
        return model
    
    def compile_model(self, model):
        """Compile a model with optimizer and loss"""
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()]
        )
        return model
    
    def get_callbacks(self, model_name):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                filepath=self.models_dir / f'{model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                self.logs_dir / f'{model_name}_training.csv',
                append=False
            ),
            TensorBoard(
                log_dir=self.logs_dir / f'{model_name}_tensorboard',
                histogram_freq=0,
                write_graph=False
            )
        ]
        return callbacks
    
    def train_model(self, model, model_name):
        """Train a single model"""
        print(f"\n🚀 Training {model_name}...")
        print(f"Epochs: {self.epochs}, Batch Size: {self.batch_size}")
        
        # Compile model
        model = self.compile_model(model)
        
        # Get callbacks
        callbacks = self.get_callbacks(model_name)
        
        # Train
        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.histories[model_name] = history
        print(f"✅ {model_name} training completed!")
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate a model on test set"""
        print(f"\n📈 Evaluating {model_name}...")
        
        # Basic evaluation
        results = model.evaluate(self.test_dataset, verbose=0)
        
        # Get predictions
        y_true = []
        y_pred_probs = []
        
        for images, labels in self.test_dataset:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            predictions = model.predict(images, verbose=0)
            y_pred_probs.extend(predictions)
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Get classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Store evaluation
        evaluation = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': results[0],
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
        
        self.evaluations[model_name] = evaluation
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Loss:      {results[0]:.4f}")
        
        return evaluation
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{model_name}_training_history.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, evaluation, model_name):
        """Plot confusion matrix"""
        cm = evaluation['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{model_name}_confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self):
        """Create comparative plots for all models"""
        if not self.evaluations:
            print("No evaluations to compare!")
            return
        
        # Prepare data
        model_names = list(self.evaluations.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison dataframe
        comparison_data = []
        for model_name in model_names:
            eval_data = self.evaluations[model_name]
            comparison_data.append({
                'Model': model_name,
                'Accuracy': eval_data['accuracy'],
                'Precision': eval_data['precision'],
                'Recall': eval_data['recall'],
                'F1-Score': eval_data['f1_score']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
            values = df_comparison[metric].values
            ax.bar(x + i*width, values, width, label=metric)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j + i*width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Save comparison to CSV
        df_comparison.to_csv(self.reports_dir / 'model_comparison.csv', index=False)
        print(f"\n📊 Comparison saved to {self.reports_dir / 'model_comparison.csv'}")
        
        return df_comparison
    
    def generate_report(self):
        """Generate comprehensive report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.reports_dir / f'classification_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BRAIN TUMOR CLASSIFICATION - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n")
            f.write(f"Number of Classes: {self.num_classes}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary table
            f.write(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, eval_data in self.evaluations.items():
                f.write(f"{model_name:<25} "
                       f"{eval_data['accuracy']:<12.4f} "
                       f"{eval_data['precision']:<12.4f} "
                       f"{eval_data['recall']:<12.4f} "
                       f"{eval_data['f1_score']:<12.4f}\n")
            
            # Detailed reports
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED CLASSIFICATION REPORTS\n")
            f.write("=" * 80 + "\n")
            
            for model_name, eval_data in self.evaluations.items():
                f.write(f"\n\n--- {model_name} ---\n")
                f.write("-" * 40 + "\n")
                
                # Convert classification report back to string
                report = eval_data['classification_report']
                f.write("\nPer-Class Metrics:\n")
                f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
                f.write("-" * 60 + "\n")
                
                for class_name in self.class_names:
                    if class_name in report:
                        metrics = report[class_name]
                        f.write(f"{class_name:<15} "
                               f"{metrics['precision']:<12.4f} "
                               f"{metrics['recall']:<12.4f} "
                               f"{metrics['f1-score']:<12.4f} "
                               f"{int(metrics['support']):<12}\n")
                
                # Confusion Matrix
                f.write("\nConfusion Matrix:\n")
                cm = eval_data['confusion_matrix']
                f.write(f"{'':>15} " + " ".join(f"{cls[:10]:>10}" for cls in self.class_names) + "\n")
                for i, row in enumerate(cm):
                    f.write(f"{self.class_names[i]:>15} " + " ".join(f"{val:>10}" for val in row) + "\n")
        
        print(f"\n📄 Comprehensive report saved to {report_path}")
        return report_path
    
    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        print("\n" + "="*60)
        print("🧠 BRAIN TUMOR CLASSIFICATION PIPELINE")
        print("="*60)
        
        # Check dataset
        if not self.check_dataset():
            return
        
        # Load data
        self.load_data(img_size=self.img_size_default)
        
        # Define models to train
        models_to_train = [
            # ('Baseline_CNN', self.build_baseline_cnn),
            # ('VGG19_Transfer', self.build_vgg19_model),
            ('ResNet50_Transfer', self.build_resnet50_model),
            ('InceptionV3_Transfer', self.build_inception_v3_model),
        ]
        
        # Train and evaluate each model
        for model_name, build_func in models_to_train:
            print(f"\n{'='*60}")
            print(f"Processing: {model_name}")
            print(f"{'='*60}")
            
            # Build model
            model = build_func(name=model_name)
            
            # Train model
            history = self.train_model(model, model_name)
            
            # Plot training history
            self.plot_training_history(history, model_name)
            
            # Evaluate model
            evaluation = self.evaluate_model(model, model_name)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(evaluation, model_name)
            
            # Reload data with correct size for next model if needed
            if model_name == 'InceptionV3_Transfer':
                self.load_data(img_size=self.img_size_default)
        
        # Create comparison plots and report
        print("\n" + "="*60)
        print("📊 GENERATING COMPARISONS AND REPORTS")
        print("="*60)
        
        comparison_df = self.plot_model_comparison()
        report_path = self.generate_report()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print best model
        best_model = max(self.evaluations.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        
        print(f"\n📁 All outputs saved to: {self.output_dir}")
        
        return comparison_df


def main():
    """Main execution function"""
    # Initialize classifier
    classifier = BrainTumorClassifier(
        data_dir='Data',
        output_dir='outputs'
    )
    
    # Run full pipeline
    results = classifier.run_full_pipeline()
    
    return classifier, results


if __name__ == "__main__":
    classifier, results = main()
