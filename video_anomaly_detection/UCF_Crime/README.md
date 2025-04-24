# UCF Crime - Weakly Supervised Video Anomaly Detection Demo


## Introduction
This repository contains the code and resources for performing video-level anomaly detection on the UCF-Crime dataset. 



## Structure
The following is the structure of the UCF_Crime directory:

* [dataset.py](./dataset.py): contains code for loading extracted features from the UCF_Crime dataset.
* [learner.py](./learner.py): contains code to implement a weakly supervised anomaly detection model based on the Multiple Instance Learning (MIL) framework.
* [loss.py](./loss.py): contains code for a custom loss function within the MIL framework
* [main.py](./main.py): contains code for training and testing the implemented Anomaly Detection (AD) method
* [UCF-Crime_demo.ipynb](./UCF-Crime_demo.ipynb): a notebook to experiment with the implemented MIL based Anomaly Detection Model



## Contributing
If you have any ideas, enhancements, or extra methods that could enhance this repository, we encourage you to participate. Simply fork this repository, implement your modifications, and then initiate a pull request. Your contributions are highly valued and welcomed!