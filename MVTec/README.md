
# Anomaly Detection Bootcamp

This repository contains a Jupyter Notebook that demonstrates two different methods for anomaly detection: CFA Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization and Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection. The methods are implemented using the `anomalib` library in Python.

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
- [Dataset](#dataset)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

This repository is part of an anomaly detection bootcamp and provides an interactive Jupyter Notebook that showcases two powerful anomaly detection methods. The methods covered in this notebook are CFA Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization and Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

## Methods

### CFA Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

This method utilizes the coupled-hypersphere-based feature adaptation technique to adapt a model for target-oriented anomaly localization. The approach aims to detect anomalies in the input data by learning a mapping from the source domain to the target domain using hypersphere-based feature adaptation.

### Draem: A discriminatively trained reconstruction embedding for surface anomaly detection

This method leverages discriminative training techniques to learn a robust representation for detecting anomalies on surfaces. It focuses on reconstructing normal regions accurately while producing distinct and noticeable reconstructions for anomalies, enabling effective surface anomaly detection in various applications.

## Dataset

The notebook utilizes the MVTEC dataset, which is a widely used benchmark dataset for anomaly detection in images. The MVTEC dataset consists of high-resolution images containing various types of defects in different object categories. It provides a challenging environment for evaluating anomaly detection methods.

## Usage

The notebook `MVTec_demo.ipynb` provides a step-by-step guide on how to use the CFA Coupled-hypersphere-based Feature Adaptation and Fully Convolutional Cross-Scale-Flows methods for anomaly detection. It includes code snippets, explanations, and examples to help you understand and apply these methods to the MVTEC dataset.

To run the notebook, make sure you have the required dependencies installed and the MVTEC dataset downloaded. Then, open the notebook in Jupyter Notebook and execute the cells sequentially to follow along with the provided instructions.

## Gradio [![gradio-backend](https://github.com/gradio-app/gradio/actions/workflows/backend.yml/badge.svg)](https://huggingface.co/spaces/masoudpz/cfa_mvtec_test)
This project showcases an anomaly detection model based on the CFA architecture trained on the MVTEC Test dataset. The CFA model is a deep learning-based approach that can identify anomalies or defects in various object and texture categories commonly encountered in industrial inspection tasks.

Features:
- Pretrained CFA model for anomaly detection
- Interactive web-based demo using Gradio
- Easy-to-use interface for uploading test images and obtaining anomaly predictions
- Efficient inference with GPU support
- Seamless integration with Hugging Face Spaces for model sharing and deployment
  
Access the project's Hugging Face Spaces page at https://huggingface.co/spaces/masoudpz/cfa_mvtec_test.
Click on the "Open in Hugging Face Spaces" button to launch the interactive demo.
Follow the instructions on the page to upload test images and view the anomaly predictions.

## Contributing

If you have any suggestions, improvements, or additional methods that could be added to this repository, please feel free to contribute. Fork this repository, make your changes, and submit a pull request. Your contributions are greatly appreciated!
