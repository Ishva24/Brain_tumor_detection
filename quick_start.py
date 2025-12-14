"""
Quick Start Script for Brain Tumor Classification
This script provides a simplified interface for quick experimentation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🧠 Brain Tumor Classification - Quick Start")
print("=" * 50)


class QuickBrainTumorClassifier:
    """Simplified version for quick experimentation"""
    
    def __init__(self):
        self.IMG_SIZE = 224
        self.BATCH_SIZE = 32
        self.EPOCHS = 10  # Reduced for quick testing
        self.data_dir = Path("Data")
        
    def load_and_prepare_data(self):
        """Load data quickly"""
        print("\n📥 Loading data...")
        
        # Check if data exists
        if not (self.data_dir / "Training").exists():
            print("⚠️ Data folder not found!")
            print("Please download the dataset from:")
            print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            print("\nOr run: python setup_project.py")
            return None, None, None
        
        # Load data
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir / "Training",
            image_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            label_mode='categorical',
            seed=42,
            validation_split=0.2,
            subset="training"
        )
        
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir / "Training",
            image_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            label_mode='categorical',
            seed=42,
            validation_split=0.2,
            subset="validation"
        )
        
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir / "Testing",
            image_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            label_mode='categorical',
            shuffle=False
        )
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        self.class_names = train_dataset.class_names
        print(f"✅ Data loaded! Classes: {self.class_names}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_simple_cnn(self):
        """Create a simple CNN model"""
        model = tf.keras.Sequential([
            layers.Rescaling(1./255, input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)),
            
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self):
        """Create VGG19 transfer learning model"""
        # Load pre-trained VGG19
        base_model = VGG19(
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Create new model
        inputs = keras.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        x = tf.keras.applications.vgg19.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(4, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def train_and_evaluate(self, model, train_data, val_data, test_data, model_name="Model"):
        """Train and evaluate a model"""
        print(f"\n🚀 Training {model_name}...")
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.EPOCHS,
            verbose=1
        )
        
        # Evaluate
        print(f"\n📊 Evaluating {model_name}...")
        test_loss, test_acc = model.evaluate(test_data, verbose=0)
        print(f"{model_name} Test Accuracy: {test_acc:.4f}")
        
        # Get predictions for detailed metrics
        y_true = []
        y_pred = []
        
        for images, labels in test_data:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        
        # Classification report
        print(f"\n📋 Classification Report for {model_name}:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Plot results
        self.plot_results(history, y_true, y_pred, model_name)
        
        return history, test_acc
    
    def plot_results(self, history, y_true, y_pred, model_name):
        """Plot training history and confusion matrix"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Training accuracy
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training loss
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[2])
        axes[2].set_title(f'{model_name} - Confusion Matrix')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_results.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def run(self):
        """Run the quick classification pipeline"""
        # Load data
        train_data, val_data, test_data = self.load_and_prepare_data()
        
        if train_data is None:
            return
        
        results = {}
        
        # Train Simple CNN
        print("\n" + "="*50)
        print("1. SIMPLE CNN MODEL")
        print("="*50)
        simple_model = self.create_simple_cnn()
        history_cnn, acc_cnn = self.train_and_evaluate(
            simple_model, train_data, val_data, test_data, "Simple_CNN"
        )
        results['Simple_CNN'] = acc_cnn
        
        # Train Transfer Learning Model
        print("\n" + "="*50)
        print("2. VGG19 TRANSFER LEARNING MODEL")
        print("="*50)
        transfer_model = self.create_transfer_learning_model()
        history_transfer, acc_transfer = self.train_and_evaluate(
            transfer_model, train_data, val_data, test_data, "VGG19_Transfer"
        )
        results['VGG19_Transfer'] = acc_transfer
        
        # Summary
        print("\n" + "="*50)
        print("📊 FINAL RESULTS SUMMARY")
        print("="*50)
        for model_name, accuracy in results.items():
            print(f"{model_name:20} : {accuracy:.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Best Model: {best_model[0]} with accuracy {best_model[1]:.4f}")
        
        return results


def main():
    """Main function"""
    classifier = QuickBrainTumorClassifier()
    results = classifier.run()
    return results


if __name__ == "__main__":
    results = main()
