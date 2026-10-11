# Multi-Task Deep Learning for Bladder Cancer Detection: Classification and Quantitative Spatial Validation of Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains the code and methodology for a retrospective, *in silico* diagnostic accuracy and algorithmic interpretability study. It implements a unified **Multi-Task Deep Learning framework** designed to simultaneously classify and segment urothelial lesions in White-Light Cystoscopy (WLC) images. 

Crucially, this project addresses the "black box" limitation of medical AI by mathematically validating the model's internal diagnostic logic. It quantifies the spatial concordance between Explainable AI (Grad-CAM) saliency maps and expert urologist ground-truth annotations.

## 🔬 Clinical Relevance
While conventional WLC is the gold standard for bladder cancer diagnosis, it is hindered by operator-dependent variability and high false-positive rates driven by benign inflammatory lesions mimicking malignancy. Existing AI models often fail to generalize due to severe class imbalance or artifactual confounding (e.g., learning lighting glare instead of true pathophysiology). 

This multi-task approach (classification + dense pixel-wise segmentation) forces the network to learn genuine morphological features, providing interpretable, real-time spatial bounding overlays suitable for transurethral resection of bladder tumor (TURBT) decision support.

## 🗄️ Dataset & Study Design
The model is trained and evaluated on the publicly available **CystoDS Dataset**, comprising 8,067 highly curated WLC images from 160 patients.
*   **Super-Classes:** Malignant (Ta/T1, CIS), Benign Mimics (inflammation, scars), and Normal/Anatomical Background.
*   **Zero-Leakage Split:** A strict patient-level split (70% Train / 15% Val / 15% Test) was enforced via `GroupShuffleSplit` to prevent data leakage from highly correlated adjacent video frames.
*   **Spatial Ground Truth:** 768 expert-verified JSON polygonal segmentations were mapped to dense binary masks to train the U-Net decoder and validate the XAI endpoints.

## ⚙️ Architecture
*   **Encoder:** Pre-trained **ConvNeXt-Tiny** (chosen for ViT-like performance with high-FPS intraoperative inference capability).
*   **Classification Head:** Global Average Pooling to a 3-class probability distribution, optimized via Categorical Focal Loss to combat extreme imbalance.
*   **Segmentation Head:** A **U-Net Decoder** utilizing skip connections from intermediate ConvNeXt stages ($16 \times 16$, $32 \times 32$, $64 \times 64$), optimized via combined Binary Cross-Entropy (BCE) and Dice Loss. Unannotated images were dynamically masked during backpropagation.

## 📊 Key Results (Hold-Out Test Set)
*   **Global Classification Accuracy:** 96.0%
*   **Malignancy Detection:** Precision (PPV) = 90%, Recall (Sensitivity) = 79%, F1-Score = 0.84.
*   **Spatial Localization Concordance:** 
    *   Mean Intersection over Union (IoU): **0.6311** (95% CI: 0.5160–0.7462)
    *   Mean Dice Similarity Coefficient (DSC): **0.7285** (95% CI: 0.6182–0.8388)
*   *Limitation Note:* The model struggled to isolate "Benign Mimics" (F1 = 0.12) due to severe data starvation (only 74 original images) and inherent visual confounding, highlighting a critical area for future federated learning research.

## 🚀 Usage & Reproducibility

### Environment Setup
To ensure strict determinism and exact reproducibility, global random seeds are forced across Python, NumPy, and TensorFlow.

*Note: A discrete GPU (e.g., NVIDIA A100) is strongly recommended for executing the dynamic data pipelines and training loops.*

### Running the Code
The primary pipeline is contained within the provided Jupyter Notebook. It covers:
1. Automated Kaggle API dataset downloading & JSON parsing.
2. Exploratory Data Analysis (EDA) and sample visualizations.
3. Albumentations-based stochastic augmentation and tf.data pipelining.
4. Model compilation, training, and checkpointing.
5. Biostatistical evaluation (Confusion Matrices, Precision, Recall, F1, DSC, IoU).
6. Multi-panel Grad-CAM visualization generation.

## 📚 Acknowledgements & References
We gratefully acknowledge the creators of the CystoDS dataset. If you build upon this work, please ensure you cite the original dataset authors:

> Lee, T.J., Qiu, L., Long, J. et al. CystoDS: a multiclass endoscopy image dataset for artificial intelligence-assisted bladder cancer detection. *Sci Data* 13, 528 (2026). https://doi.org/10.1038/s41597-026-06887-z

## 📄 License
This project is licensed under the MIT License.
