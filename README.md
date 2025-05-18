# Leukemia Cell Classification using Attention-Based CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📑 Overview

This repository contains the implementation of an advanced deep learning approach for automatic classification of leukemic cells, as described in the paper: **"Enhanced Leukemia Cell Classification Using Attention-Based CNN and Advanced Augmentation"**. The proposed model combines the EfficientNetV2-B3 architecture with attention mechanisms to achieve state-of-the-art performance in distinguishing between normal and malignant cells in blood samples.

<p align="center">
  <img src="https://github.com/username/leukemia-classification/blob/main/images/attention_visualization.png" alt="Visualization of attention maps" width="600"/>
  <br>
  <em>Example of attention visualization from our model</em>
</p>

## 🔬 Research Context

Acute lymphoblastic leukemia (ALL) is the most common childhood cancer, with early diagnosis being crucial for successful treatment. Traditional diagnosis requires manual examination of blood smears by experts, which can be time-consuming and subject to inter-observer variability.

Our research introduces a novel attention-based convolutional neural network with Squeeze-and-Excitation mechanisms to classify leukemic cells with high accuracy. The model achieves an F1-score of **95.55%** on the C-NMC 2019 dataset, outperforming previous approaches including VGG16, VGG19, and Xception architectures.

## 💻 Technology Stack

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **torchvision**: For image transformations and pre-trained models
- **OpenCV**: For image processing and feature extraction
- **scikit-learn**: For evaluation metrics and traditional ML models
- **NumPy/Pandas**: For data handling and analysis
- **Matplotlib/Seaborn**: For visualization
- **timm**: PyTorch Image Models for advanced architectures

## 🏗️ Architecture

Our model architecture consists of:

1. **Backbone**: Modified EfficientNetV2-B3 pre-trained on ImageNet
2. **Attention Mechanism**: Squeeze-and-Excitation blocks for channel-wise feature recalibration
3. **Classification Head**: Multi-layer network with dropout for regularization
4. **Loss Function**: Focal Loss to address class imbalance

## 🗂️ Repository Structure

```
├── dataset/               # Dataset organization scripts
├── models/                # Model architecture definitions
│   ├── attention.py       # Attention mechanisms implementation
│   ├── efficientnet.py    # Modified EfficientNet backbone
│   └── hybrid_model.py    # Hybrid model for feature fusion
├── features/              # Feature extraction modules
├── utils/                 # Utility functions
│   ├── augmentation.py    # Advanced augmentation pipeline
│   ├── evaluation.py      # Metrics and evaluation tools
│   └── visualization.py   # Attention map visualization
├── train.py               # Training script
├── test.py                # Evaluation script
├── inference.py           # Inference on new images
└── notebooks/             # Jupyter notebooks for experiments
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py --data_dir path/to/dataset --batch_size 16 --epochs 50 --lr 3e-5
```

### Evaluation

```bash
python test.py --model_path path/to/model.pt --data_dir path/to/test_data
```

### Inference

```bash
python inference.py --image path/to/image.jpg --model_path path/to/model.pt
```

## 📊 Results

| Model                 | F1-Score | Accuracy | AUC     |
|-----------------------|----------|----------|---------|
| Our Attention-Based CNN | 95.55%   | 95.53%   | 98.14%  |
| VGG16                 | 92.60%   | 92.35%   | 96.80%  |
| VGG19                 | 91.75%   | 91.70%   | 96.10%  |
| Xception              | 90.76%   | 90.80%   | 95.55%  |

## 👥 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{braga2023enhanced,
  title={Enhanced Leukemia Cell Classification Using Attention-Based CNN and Advanced Augmentation},
  author={Braga, Douglas Costa and Dantas, Daniel Oliveira},
  journal={Name of Journal/Conference},
  year={2023}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- We acknowledge the SBILab research team for providing the C-NMC 2019 dataset.
- We thank the Universidade Federal de Sergipe for supporting this research.

