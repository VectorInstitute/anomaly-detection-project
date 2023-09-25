# Anomaly Detection on UCF-Crime Dataset

## Introduction
This repository contains the code and resources for performing video-level anomaly detection on the UCF-Crime dataset. 

## Dataset
One of the most famouse large-scale dataset video anomaly detection dataset with video-level labels is [UCF-crime](https://www.crcv.ucf.edu/projects/real-world/) dataset that contains 1,900 untrimmed real-world outdoor and indoor surveillance videos. The total length of the videos is 128 hours, which contains 13 classes of anomalous events including: 1. Abuse, 2. Arrest, 3. Arson, 4. Assault, 5. Burglary, 6. Explosion, 7. Fighting, 8. Road Accident, 9. Robbery, 10. Shooting, 11. Stealing, 12. Shoplifting, 13. Vandalism.

## Methodology
We utilized a two-stream Inflated 3D (I3D) Convolutional Network to extract RGB and Flow features from the video. The RGB stream extracts information related to the appearance of objects and scenes, while the optical Flow stream captures the motion and dynamics of objects between frames. We then combined
the information from both streams by concatenating the learned RGB and Flow features. This provides a more complete understanding of the video content, leading to improved anomaly detection accuracy.

To avoid annotating abnormal activities in training videos, a weakly supervised anomaly detection
model was implemented based on the Multiple Instance Learning (MIL) framework. The
model considers normal and abnormal videos as bags and video clips as instances. It learns a
ranking model to predict high anomaly scores for video clips containing anomalies.


![proposedmethod](https://github.com/VectorInstitute/anomaly-detection-project/assets/23232055/ff15b1e8-00e5-403a-bff9-3426d8acb7a4)

## Structure
The following is the structure of the UCF-Crime directory:

* [dataset.py](./UCF-Crime/dataset.py): contains code for loading extracted features from the UCF-Crime dataset.
* [learner.py](./UCF-Crime/learner.py): contains code to implement a weakly supervised anomaly detection model based on the Multiple Instance Learning (MIL) framework.
* [loss.py](./UCF-Crime/loss.py): contains code for a custom loss function within the MIL framework
* [main.py](./UCF-Crime/main.py): contains code for training and testing the implemented anomaly detection (AD) method
<!-- * [UCF-Crime_demo.ipynb](./UCSDPedestraian/UCSDPedestrain_demo.ipynb): a notebook to experiment with MIL algorithm -->



## Contributing
If you have any ideas, enhancements, or extra methods that could enhance this repository, we encourage you to participate. Simply fork this repository, implement your modifications, and then initiate a pull request. Your contributions are highly valued and welcomed!