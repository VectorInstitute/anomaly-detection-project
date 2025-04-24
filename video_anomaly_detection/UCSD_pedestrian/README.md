# UCSD Pedestrain - Unsupervised Video Anomaly Detection Demo

## Introduction

This repository contains codes to run the frame-level video anomaly detection method on the UCSD pedestrain dataset. UCSD pedestrain was collected in a spacious walkway where pedestrians moved parallel to the camera plane using a stationary camera.


## Structure
The following is the directory structure of the UCSDPedestrain directory:


* [DMAD/](./UCSDPedestrian/DMAD) ==> contains codes for diversity measurable anomaly detection method
  * [model.py](./UCSDPedestrian/DMAD/model.py) ==> contains codes for DMAD model
  * [model_config.yaml](./UCSDPedestrian/DMAD/model_config.yaml) ==> contains config to run DMAD algorithm
  * [run.py](./UCSDPedestrian/DMAD/run.py) ==> contains codes to run DMAD algorithm
* [Dataset/](./UCSDPedestrian/Dataset) ==> contains codes to prepare the UCSD dataset
  * [UCSD_dataset.py](./UCSDPedestrian/Dataset/UCSD_dataset.py) ==> contains codes to get the train and test dataset
  * [data_config.yaml](./UCSDPedestrian/Dataset/data_config.yaml) ==> contains config to get UCSD dataset
* [Demo_run.py](./UCSDPedestrian/Demo_run.py) ==> a python file to run and evaluate DMAD algorithm from terminal
* [UCSDPedestrain_demo.ipynb](./UCSDPedestrian/UCSDPedestrian_demo.ipynb) ==> a notebook to experiment with DMAD algorithm
* [utils.py](./UCSDPedestrian/utils.py) ==> contains some utility functions




## Usage

`UCSDPedestrian_demo.ipynb` is a notebook to run this algorithm. You can also run `Demo_run.py` file using following command to run this algorithm from terminal.

```bash
python Demo_run.py 
```

## Contributing 

If you have any suggestions, improvements, or additional methods that could be added to this repository, please feel free to contribute. Fork this repository, make your changes, and submit a pull request. Your contributions are greatly appreciated!
