# 🫀 Myocardial Fibrosis Detection from CT Images

Author: Vlad-Nicolae Căpraru
Academic Year: 2024–2025
University: Technical University of Cluj-Napoca


# 📘 Project Overview

This project presents a semi-automatic system for detecting myocardial fibrosis in cardiac CT images, using image processing and deep learning techniques.
The approach combines Hounsfield Unit (HU) filtering with the YOLOv8 model to segment and identify fibrotic regions of the myocardium.


# 🎯 Objectives

Generate binary masks for heart and fibrosis regions.

Automatically segment the heart from CT images.

Prepare a dataset for model training.

Train a YOLOv8 model for fibrosis classification (150 epochs).

Evaluate model performance on test data.

Develop a Python program for fibrosis detection.

Validate and document the obtained results.


# ⚙️ Technical Specifications

CT Image Format: DICOM → converted to PNG for training.

Image Resolution: 512×512 (normalized intensity).

Detection range: 60–90 HU (fibrosis threshold).

Tools: Python 3.12, YOLOv8, OpenCV, NumPy, Pandas, Matplotlib, PyDicom.

Hardware: Windows 11, NVIDIA RTX 3060, 32 GB RAM.


# 🧠 Methodology

Segmentation:

Used YOLOv8 to localize potential fibrosis zones.

HU Filtering:

Identified fibrotic tissue using pixel values between 60–90 HU.

Computation:

Calculated fibrosis percentage per image and exported results to Excel.

Validation:

Compared model output with manual visual assessment on 36 anonymized CT scans.


# 🧩 Implementation

Entirely implemented in Python, running in terminal / VS Code / PyCharm.

Modular structure with separate functions for:

Reading DICOM images

Converting to HU and PNG

Running YOLO inference

Calculating fibrosis percentage

Saving outputs and summaries (Excel)


# 📊 Results

Accurate detection for all manually verified fibrosis cases.

False-positive rate below 5%.

Fibrosis percentage range: 2%–35% per image.

Successful integration of YOLOv8 with HU filtering.

Output data exported in .xlsx format for future analysis.


# 🧾 Conclusions

Demonstrated feasibility of using CT-based AI models for fibrosis detection as a non-invasive and cost-effective alternative to MRI.

Built a hybrid HU + AI system capable of identifying myocardial fibrosis semi-automatically.

The system is modular, reusable, and adaptable for future clinical integration.


# 🚀 Future Work

Multi-class fibrosis classification (mild, moderate, severe).

Graphical user interface (GUI) for medical use.

Clinical validation with real patient datasets.

Integration into web or desktop applications.


# 🔬 References

Selected references include:

Goulin et al., Computers in Biology and Medicine, 2022 – Deep learning-based cardiac segmentation.

Wang et al., Nature Communications, 2024 – MedSAM model for universal medical segmentation.

Penso et al., Frontiers in Cardiovascular Medicine, 2023 – Deep learning approach for fibrosis detection in CT.

Zhang et al., MDPI, 2024 – FibrosisNet for MRI-based fibrosis classification.
