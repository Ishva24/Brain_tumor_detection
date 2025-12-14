"""
Setup script for Brain Tumor Classification Project
This script helps with dataset download and environment setup
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
import shutil


def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "tensorflow>=2.10.0",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pillow",
        "opencv-python",
        "kaggle"  # For dataset download
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("✅ All packages installed successfully!")


def download_dataset_kaggle():
    """Download dataset from Kaggle (requires Kaggle API key)"""
    print("\n📥 Downloading dataset from Kaggle...")
    print("Note: This requires Kaggle API credentials.")
    print("If you don't have them, download manually from:")
    print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'masoudnickparvar/brain-tumor-mri-dataset',
            path='.',
            unzip=True
        )
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nPlease download the dataset manually from:")
        print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        return False


def prepare_sample_data():
    """Create sample data structure for testing"""
    print("\n🎯 Creating sample data structure for testing...")
    
    # Create directory structure
    base_dir = Path("Data")
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    for split in ['Training', 'Testing']:
        for cls in classes:
            dir_path = base_dir / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy file to show structure
            dummy_file = dir_path / "sample.txt"
            dummy_file.write_text(f"Place {cls} MRI images here (.jpg or .png)")
    
    print("✅ Sample structure created in 'Data' folder")
    print("\n📁 Expected structure:")
    print("Data/")
    print("├── Training/")
    print("│   ├── glioma/")
    print("│   ├── meningioma/")
    print("│   ├── notumor/")
    print("│   └── pituitary/")
    print("└── Testing/")
    print("    ├── glioma/")
    print("    ├── meningioma/")
    print("    ├── notumor/")
    print("    └── pituitary/")


def main():
    """Main setup function"""
    print("="*60)
    print("🧠 BRAIN TUMOR CLASSIFICATION - SETUP")
    print("="*60)
    
    # Install requirements
    response = input("\n1. Install Python packages? (y/n): ").lower()
    if response == 'y':
        install_requirements()
    
    # Download dataset
    response = input("\n2. Try to download dataset from Kaggle? (y/n): ").lower()
    if response == 'y':
        success = download_dataset_kaggle()
        if not success:
            # Create sample structure
            response = input("\n3. Create sample folder structure? (y/n): ").lower()
            if response == 'y':
                prepare_sample_data()
    else:
        # Create sample structure
        response = input("\n3. Create sample folder structure? (y/n): ").lower()
        if response == 'y':
            prepare_sample_data()
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Ensure dataset is in 'Data' folder with correct structure")
    print("2. Run: python brain_tumor_classifier.py")
    print("3. Check 'outputs' folder for results")


if __name__ == "__main__":
    main()
