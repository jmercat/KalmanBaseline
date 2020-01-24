# Baseline Models

This is a code for baseline production using Kalman filter.
It is the implementation of the models presented in : https://arxiv.org/abs/1908.11472

To use this code :
* Set the parameters, dataset path settings.yaml (The Bicycle model may show training unstabilities, contributions are welcome.)
* Run train_kalman_predict.py starts the trainning.
* Enter the name of the trained model in the load_name field of settings.yaml (should be in the form \<model>\_\<dataset>\_\<id>) 
* Run plot_results.py to plot trajectory samples, estimated position and predictions
* Run save_results.py to save the prediction results computed on the test set
* Run stats_results.py to print metrics evaluation, plot covariance matching and error histogram (from the saved results)

#Datasets

## NGSIM
  
From NGSIM website:  
* Register at https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj  
* Download US-101-LosAngeles-CA.zip and I-80-Emeryville-CA.zip  
* Unzip and extract vehicle-trajectory-data into raw/us-101 and raw/i-80  
  
From googledrive:  
* Download i-80: https://drive.google.com/open?id=19ovxiJLCnS1ar1dYvUIrfdz_UnGejK-k  
* Download us-101: https://drive.google.com/open?id=14dMKew22_5evfOoSGEBYqHP3L92e7nyJ  

Dataset fields:  
* doc/trajectory-data-dictionary.htm  

This Dataset is to be pre-processed with the Matlab function preprocess_data.m (that is a slightly modified version of the one from https://github.com/nachiket92/conv-social-pooling)

* Already preprocessed samples:
    * https://drive.google.com/open?id=171YZvK2DnJbtIgve6qAOkk6AFNrGAYTJ
    * https://drive.google.com/open?id=1N080iiQc43MTvLMqaLtsZh49okJL0hPx
    
    
## Argoverse

Get the data from the Argoverse website:
* https://www.argoverse.org/

Put their API from Argoverse in a directory at same depth named "Argoverse"
* https://github.com/argoai/argoverse-api

## Fusion

The Fusion dataset option relates to a private dataset that is not available.
