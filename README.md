# Baseline Models

This is a code for baseline production using Kalman filter.
It is the implementation of the models presented in : https://arxiv.org/abs/1908.11472

This may be quiet slow to train... contributions are welcome

Kalman filter approximated trained parameters are hard coded as initial
values.

# NGSIM dataset
  
From NGSIM website:  
* Register at https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj  
* Download US-101-LosAngeles-CA.zip and I-80-Emeryville-CA.zip  
* Unzip and extract vehicle-trajectory-data into raw/us-101 and raw/i-80  
  
From googledrive:  
* Download i-80: https://drive.google.com/open?id=19ovxiJLCnS1ar1dYvUIrfdz_UnGejK-k  
* DOwnload us-101: https://drive.google.com/open?id=14dMKew22_5evfOoSGEBYqHP3L92e7nyJ  
  
Dataset fields:  
* doc/trajectory-data-dictionary.htm  

# Reference .mat files
Obtained with preprocess_data.m applied to above NGSIM dataset    

