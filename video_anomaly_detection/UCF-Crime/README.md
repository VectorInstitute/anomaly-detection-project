# UCF-Crime Anomaly Detection Bootcamp


## Introduction
This repository contains the code and resources for performing video-level anomaly detection on the UCF-Crime dataset. 

## Dataset
One of the most famouse large-scale dataset video anomaly detection dataset with video-level labels is [UCF-crime](https://www.crcv.ucf.edu/projects/real-world/) dataset that contains 1,900 untrimmed real-world outdoor and indoor surveillance videos. The total length of the videos is 128 hours, which contains 13 classes of anomalous events including: 1. Abuse, 2. Arrest, 3. Arson, 4. Assault, 5. Burglary, 6. Explosion, 7. Fighting, 8. Road Accident, 9. Robbery, 10. Shooting, 11. Stealing, 12. Shoplifting, 13. Vandalism.


## Structure
The following is the structure of the UCF-Crime directory:

* [dataset.py](./UCF-Crime/dataset.py): contains code for loading extracted features from the UCF-Crime dataset.
* [learner.py](./UCF-Crime/learner.py): contains code to implement a weakly supervised anomaly detection model based on the Multiple Instance Learning (MIL) framework.
* [loss.py](./UCF-Crime/loss.py): contains code for a custom loss function within the MIL framework
* [main.py](./UCF-Crime/main.py): contains code for training and testing the implemented anomaly detection (AD) method
* [UCF-Crime_demo.ipynb](./UCF-Crime/UCF-Crime_demo.ipynb): a notebook to experiment with the implemented MIL based Anomaly Detection Model
<!-- * [UCF-Crime_demo.ipynb](./UCSDPedestraian/UCSDPedestrain_demo.ipynb): a notebook to experiment with MIL algorithm -->



## Contributing
If you have any ideas, enhancements, or extra methods that could enhance this repository, we encourage you to participate. Simply fork this repository, implement your modifications, and then initiate a pull request. Your contributions are highly valued and welcomed!