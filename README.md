# Anomaly Detection Bootcamp

This repository contains the resources and code for an Anomaly Detection Bootcamp, focused on exploring and implementing anomaly detection techniques. Anomaly detection plays a crucial role in various domains, including fraud detection, network intrusion detection, system monitoring, and image and video analysis. The bootcamp aims to provide participants with a comprehensive understanding of anomaly detection algorithms and their practical applications.

## Introduction to Anomaly Detection

Anomaly detection is the process of identifying patterns or instances that deviate significantly from the expected behavior within a dataset. These anomalies can be indicative of critical events, errors, or malicious activities, making anomaly detection an essential task in various real-world scenarios. By leveraging machine learning techniques, anomaly detection algorithms strive to identify and flag anomalous instances accurately.

## Datasets

The bootcamp utilizes four diverse datasets to cover a broad range of anomaly detection applications. Each dataset offers unique challenges and opportunities for participants to explore and experiment with different techniques.

### MVTec AD

The MVTec AD dataset serves as a benchmark for anomaly detection and segmentation in images. It consists of 5,000 high-resolution images, categorized into fifteen different object and texture categories. Each category comprises a set of normal training images and a test set containing images with various defects, as well as defect-free images. Participants will work with this dataset to develop image-based anomaly detection models.

### Bank Account Fraud (BAF)

The Bank Account Fraud (BAF) suite is derived from a real-world online bank account opening fraud detection dataset. It consists of six individual tabular datasets, each featuring predetermined and controlled types of data bias across multiple time-steps. The BAF suite emphasizes the importance of fair ML practices in the context of financial services, considering the potential impact of model predictions on individuals' access to financial resources and social equity.

Each set in the BAF datasets consist of one million instances with thirty features, including protected attributes such as age, personal income, and employment status. To address privacy concerns, state-of-the-art Generative Adversarial Network (GAN) models were employed to generate the datasets, with injected differential privacy into the instances of the original dataset. The BAF datasets are designed to stress test the performance and fairness of ML models in dynamic environments.

### DGraph

DGraph is a large-scale graph dataset specifically focused on fraud detection in real-world finance scenarios. The dataset is provided by Finvolution Group, a prominent player in China's online consumer finance industry. Each user's raw data in DGraph includes five components: user ID, basic personal profile information, telephone number, borrowing behavior, and emergency contacts.

DGraph consists of three million nodes in a graph, with over one million highly imbalanced ground-truth nodes considered anomalous. Only 1.3% of the nodes are considered anomalous, while the remaining two million users are considered background nodes. Each node represents a user and has a 17-dimensional feature vector derived from their personal information. The presence of an edge between two users signifies that one user regards the other as their emergency contact. Experiments in DGraph are conducted with different ratios of labeled anomalies to evaluate the effectiveness of anomaly detection algorithms.

### UCSD Pedestrian

The UCSD Anomaly Detection dataset is created by utilizing a fixed camera placed at an elevated position for observation, providing an overview of pedestrian walkways. The dataset was collected in a spacious walkway where pedestrians moved parallel to the camera plane. Across these walkways, the density of the crowd varied from sparse to becoming highly congested.

The dataset comprises 16 training video samples and 12 testing video samples, each containing approximately 200 frames. The training videos contain only normal instances, while the testing videos contain both normal and anomalous instances. The anomalous instances include people running, bicycling, skating, and walking in groups. The dataset is designed to evaluate the performance of anomaly detection algorithms in crowded scenes.

### UCF-Crime
One of the most famouse large-scale dataset video anomaly detection dataset with video-level labels is UCF-crime dataset. The total length of the videos is 128 hours, which contains 13 classes of anomalous events including: 1. Abuse, 2. Arrest, 3. Arson, 4. Assault, 5. Burglary, 6. Explosion, 7. Fighting, 8. Road Accident, 9. Robbery, 10. Shooting, 11. Stealing, 12. Shoplifting, 13. Vandalism.

The UCF-Crime dataset consists of 1900 surveillance videos, with the training set consisting of 800 normal videos and 810 abnormal videos. The testing set includes 150 normal videos and 140 anomaly videos, covering all 13 anomaly categories in both sets. 

## Getting Started

To get started with the bootcamp and access the datasets, please follow the instructions provided in the respective dataset folders. Each dataset folder contains the necessary documentation, code examples, and data files required for the bootcamp sessions.

We hope this bootcamp provides a valuable learning experience in the exciting field of anomaly detection. Feel free to explore, experiment, and contribute to this repository. If you have any questions or feedback, please don't hesitate to reach out.
