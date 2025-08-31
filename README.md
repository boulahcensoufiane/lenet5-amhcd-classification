LeNet-5 AMHCD Classification
A complete PyTorch implementation of the LeNet-5 architecture for Amazigh Handwritten Character Recognition using the AMHCD (Amazigh Handwritten Character Database) dataset.

📋 Project Overview
This repository contains a PyTorch implementation of the LeNet-5 convolutional neural network for classifying Amazigh handwritten characters from the AMHCD dataset. The project includes training scripts, evaluation metrics, visualization tools, and comprehensive documentation for reproducible research.

The implementation achieves 91.23% accuracy on the test set with Macro F1 score of 0.910 and Weighted F1 score of 0.912.

🗂️ Dataset Information
The Amazigh Handwritten Character Database (AMHCD) contains 33 classes of Amazigh characters with approximately 9,600 total samples. The dataset follows this structure:

text
data/amhcd/
├── class_0/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class_1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── ...
Dataset Source: AMHCD on Kaggle

⚙️ Installation
Prerequisites
Python 3.9 or higher

pip package manager

Setup
Clone the repository:

bash
git clone https://github.com/yourusername/lenet5-amhcd-classification.git
cd lenet5-amhcd-classification
Install dependencies:

bash
pip install -r requirements.txt
Required Packages
The project requires the following Python packages:

text
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
umap-learn>=0.5.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.21.0
jupyter>=1.0.0
🚀 Usage
Training the Model
To train the LeNet-5 model on the AMHCD dataset:

bash
python src/train.py --data_dir data/amhcd --epochs 30 --batch_size 128
Arguments:

--data_dir: Path to the AMHCD dataset (default: data/amhcd)

--epochs: Number of training epochs (default: 30)

--batch_size: Batch size for training (default: 128)

--lr: Learning rate (default: 0.001)

--seed: Random seed (default: 42)

Evaluation and Visualization
To evaluate the trained model and generate visualizations:

bash
python src/eval.py --model_path models/best.pt --data_dir data/amhcd
This script will:

Evaluate model performance on the test set

Generate training curves

Create a confusion matrix

Produce Grad-CAM visualizations

Generate UMAP feature visualizations

📊 Results
The implemented LeNet-5 model achieves the following performance on the AMHCD test set:

Metric	Value
Accuracy	91.23%
Macro F1	0.910
Weighted F1	0.912
