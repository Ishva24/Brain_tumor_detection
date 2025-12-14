"""
Inference Script for Brain Tumor Classification
Use this script to make predictions on new MRI images using trained models
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image


class BrainTumorPredictor:
    """Class for making predictions on new brain MRI images"""
    
    def __init__(self, model_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model file (.keras or .h5)
        """
        self.model = None
        self.model_path = model_path
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.img_size = 224  # Default size, will be adjusted based on model
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            print(f"Loading model from {model_path}...")
            self.model = keras.models.load_model(model_path)
            
            # Get input shape
            input_shape = self.model.input_shape
            if input_shape:
                self.img_size = input_shape[1]  # Assuming square images
            
            print(f"Model loaded successfully!")
            print(f"   Input size: {self.img_size}x{self.img_size}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Resize
        image = image.resize((self.img_size, self.img_size))
        
        # Convert to array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Note: Model should have its own preprocessing layer
        # If using transfer learning models, might need specific preprocessing
        
        return img_array
    
    def predict_single_image(self, image_path, show_plot=True):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image
            show_plot: Whether to show visualization
        
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            print("No model loaded! Please load a model first.")
            return None
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        original_image = Image.open(image_path).convert('RGB')
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Create results dictionary
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }
        
        # Visualization
        if show_plot:
            self.visualize_prediction(original_image, results)
        
        return results
    
    def visualize_prediction(self, image, results):
        """Visualize prediction results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title(f"Predicted: {results['predicted_class']}\n"
                         f"Confidence: {results['confidence']:.2%}")
        
        # Show probability distribution
        probs = list(results['all_probabilities'].values())
        classes = list(results['all_probabilities'].keys())
        colors = ['red' if c == results['predicted_class'] else 'blue' for c in classes]
        
        axes[1].bar(classes, probs, color=colors)
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Class Probabilities')
        axes[1].set_ylim(0, 1)
        
        # Add percentage labels on bars
        for i, (c, p) in enumerate(zip(classes, probs)):
            axes[1].text(i, p + 0.01, f'{p:.1%}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, image_folder, output_csv=None):
        """
        Make predictions on multiple images in a folder
        
        Args:
            image_folder: Path to folder containing images
            output_csv: Optional path to save results as CSV
        
        Returns:
            List of prediction results
        """
        if self.model is None:
            print("No model loaded! Please load a model first.")
            return None
        
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob('*.jpg')) + \
                     list(image_folder.glob('*.jpeg')) + \
                     list(image_folder.glob('*.png'))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return None
        
        print(f"Making predictions on {len(image_files)} images...")
        
        results = []
        for img_path in image_files:
            # Make prediction
            pred = self.predict_single_image(img_path, show_plot=False)
            
            # Add to results
            results.append({
                'filename': img_path.name,
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence'],
                **{f'prob_{cls}': prob for cls, prob in pred['all_probabilities'].items()}
            })
            
            print(f"  {img_path.name}: {pred['predicted_class']} ({pred['confidence']:.2%})")
        
        # Save to CSV if requested
        if output_csv:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Results saved to {output_csv}")
        
        return results
    
    def evaluate_on_test_set(self, test_dir):
        """
        Evaluate model on a test set with known labels
        
        Args:
            test_dir: Directory with subdirectories for each class
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            print("No model loaded! Please load a model first.")
            return None
        
        test_dir = Path(test_dir)
        
        y_true = []
        y_pred = []
        
        print("Evaluating on test set...")
        
        for class_name in self.class_names:
            class_dir = test_dir / class_name
            if not class_dir.exists():
                print(f"  Warning: {class_dir} not found")
                continue
            
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                # Make prediction
                pred = self.predict_single_image(img_path, show_plot=False)
                
                # Store results
                y_true.append(class_name)
                y_pred.append(pred['predicted_class'])
        
        # Calculate metrics
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }


def demo_prediction():
    """Demo function showing how to use the predictor"""
    print("="*60)
    print("BRAIN TUMOR PREDICTION DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = BrainTumorPredictor()
    
    # Check for existing models
    model_dir = Path("outputs/models")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
        
        if model_files:
            print(f"\nFound {len(model_files)} saved models:")
            for i, model_file in enumerate(model_files, 1):
                print(f"  {i}. {model_file.name}")
            
            # Load the first model as example
            predictor.load_model(model_files[0])
            
            # Demo: Create a dummy image for testing
            print("\nCreating demo image for testing...")
            demo_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            demo_image = Image.fromarray(demo_image)
            
            # Save demo image
            demo_path = "demo_mri.jpg"
            demo_image.save(demo_path)
            
            # Make prediction
            print("\nMaking prediction on demo image...")
            results = predictor.predict_single_image(demo_path)
            
            print("\nPrediction Results:")
            print(f"  Predicted Class: {results['predicted_class']}")
            print(f"  Confidence: {results['confidence']:.2%}")
            print("\n  All Probabilities:")
            for cls, prob in results['all_probabilities'].items():
                print(f"    {cls}: {prob:.4f}")
            
            # Clean up
            os.remove(demo_path)
            
        else:
            print("\nNo trained models found in outputs/models/")
            print("Please run the training script first:")
            print("  python brain_tumor_classifier.py")
    else:
        print("\nModels directory not found!")
        print("Please run the training script first:")
        print("  python brain_tumor_classifier.py")
    
    print("\n" + "="*60)
    print("Demo completed!")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Prediction")
    parser.add_argument("--model", type=str, help="Path to trained model file")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--folder", type=str, help="Path to folder of images for batch prediction")
    parser.add_argument("--test", type=str, help="Path to test directory for evaluation")
    parser.add_argument("--output", type=str, help="Output CSV file for batch predictions")
    parser.add_argument("--demo", action="store_true", help="Run demo prediction")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_prediction()
        return
    
    # Initialize predictor
    predictor = BrainTumorPredictor()
    
    # Load model
    if args.model:
        predictor.load_model(args.model)
    else:
        print("Please specify a model with --model")
        return
    
    # Make predictions
    if args.image:
        results = predictor.predict_single_image(args.image)
        print("\n📊 Results:")
        print(f"Predicted: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2%}")
    
    elif args.folder:
        results = predictor.predict_batch(args.folder, args.output)
    
    elif args.test:
        results = predictor.evaluate_on_test_set(args.test)
    
    else:
        print("Please specify --image, --folder, or --test")


if __name__ == "__main__":
    main()
