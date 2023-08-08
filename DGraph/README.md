# DGraph Anomaly Detection Bootcamp

## Introduction

This repository contains codes to run various anomaly detection algorithms on the DGraph dataset. DGraphFin is a new large scale graph dataset introduced in NeurIPS 2022 conference and sourced from real-world finance scenarios, specifically focused on fraud detection. 

We study 4 catagories of algorithms on DGraph dataset with various ratio of labeled anomalies. The 4 catagories are:
1) Supervised anomaly detection algorithms on graph data: **AMNet**, **GCN** and **SAGE**
2) Unsupervised anomaly detection algorithms on graph data: **OCGNN**, **DONE** and **AdONE**
3) Supervised anomaly detection algorithms on tabular data: **MLP**, **FTTransformter**, **DeepSAD** and **XGBoost**
4) Unsupervised anomaly detection algorithms on tabular data: **IForest** and **CBLOF**

## Structure
The following is the directory structure of the DGraph directory:

* [Baselines/](./Baselines) ==> contains codes for various anomaly detection baselines
  * [GraphSupervised/](./Baselines/GraphSupervised) ==> codes for supervised anomaly detection algorithms on graph data
    * [AMNet/](./Baselines/GraphSupervised/AMNet) ==> codes for **AMNet** algorithm
      * [models/](./Baselines/GraphSupervised/AMNet/models) ==> contains codes for **AMNet** model
      * [run.py](./Baselines/GraphSupervised/AMNet/run.py) ==> contains codes to run **AMNet** algorithm
    * [GNN/](./Baselines/GraphSupervised/GNN) ==> codes for **GCN** and **SAGE** anomaly detection algorithms
      * [models/](./Baselines/GraphSupervised/GNN/models) ==> contains codes for **GCN** and **SAGE** models
      * [run.py](./Baselines/GraphSupervised/GNN/run.py) ==> contains codes to run **GCN** and **SAGE** algorithms
  * [GraphUnsupervised/](./Baselines/GraphUnsupervised) ==> codes for unsupervised anomaly detection algorithms on graph data
    * [models/](./Baselines/GraphUnsupervised/models) ==> contains codes for **OCGNN**, **DONE** and **AdONE** models
    * [trainer/](./Baselines/GraphUnsupervised/trainer) ==> contains codes for **OCGNN**, **DONE** and **AdONE** trainers
    * [run.py](./Baselines/GraphUnsupervised/run.py)    ==> contains codes to run **OCGNN**, **DONE** and **AdONE** algorithms
  * [TabularSupervised/](./Baselines/TabularSupervised) ==> codes for supervised anomaly detection algorithms on tabular data
    * [DeepCNN/](./Baselines/TabularSupervised/DeepCNN) ==> codes for **MLP** and **FTTransformter** algorithms
      * [run.py](./Baselines/TabularSupervised/DeepCNN/run.py) ==> contains codes to run **MLP** and **FTTransformter** algorithms
    * [DeepSAD/](./Baselines/TabularSupervised/DeepSAD) ==> codes for **DeepSAD** algorithm
      * [data_loader/](./Baselines/TabularSupervised/DeepSAD/data_loader) ==> contains codes for **DeepSAD** data loader
      * [models/](./Baselines/TabularSupervised/DeepSAD/models) ==> contains codes for **DeepSAD** model
      * [trainer/](./Baselines/TabularSupervised/DeepSAD/trainer) ==> contains codes for **DeepSAD** trainer
      * [run.py](./Baselines/TabularSupervised/DeepSAD/run.py) ==> contains codes to run **DeepSAD** algorithm
    * [XGBoost/](./Baselines/TabularSupervised/XGBoost) ==> codes for **XGBoost** algorithm
      * [run.py](./Baselines/TabularSupervised/XGBoost/run.py) ==> contains codes to run **XGBoost** algorithm
  * [TabularUnsupervised/](./Baselines/TabularUnsupervised) ==> codes for unsupervised anomaly detection algorithms on tabular data
    * [run.py](./Baselines/TabularUnsupervised/run.py) ==> contains codes to run **IForest** and **CBLOF** algorithms
* [Dataset/](./Dataset) ==> contains codes to prepare the DGraph dataset
  * [dgraphfin.py](./Dataset/dgraphfin.py) ==> contains codes to get the DGraph dataset
  * [prepare_data.py](./Dataset/prepare_data.py) ==> contains codes to prepare the DGraph dataset based on the given ratio of labeled anomaly parameter
* [DGraphFin_demo.ipynb](./DGraphFin_demo.ipynb) ==> a notebook to experiment with these algorithms
* [Demo_run.py](./Demo_run.py) ==> a python file to run these algorithms from terminal
* [myutils.py](./myutils.py)  ==> contains some utility functions



## Usage

`DGraphFin_demo.ipynb` is a notebook to experiment with these algorithms. You can also run `Demo_run.py` file using following command to run this algorithms from terminal.

```bash
python Demo_run.py 
\ --model_type ['Graph'/'Tabular'] 
\ --supervision ['Supervised'/'Unsupervise'] 
\ --model_name [model_name] 
\ --dir [directory to save results] 
\ --suffix [suffix to add to the result file name] 
\ --seed_list [list of seed to run] 
\ --rla_list [list of ratio of labeled anomalies]
```
 In this files, each of these algorithms are first imported from their corresponding `run.py` file and then run on the DGraph dataset. The results are saved in the given directory with specified suffix. You can also specify seed list and list of ratio of labeled anomalies to run these algorithms multiple times with different seeds and different ratio of labeled anomalies.

## Contributing 

If you have any suggestions, improvements, or additional methods that could be added to this repository, please feel free to contribute. Fork this repository, make your changes, and submit a pull request. Your contributions are greatly appreciated!
