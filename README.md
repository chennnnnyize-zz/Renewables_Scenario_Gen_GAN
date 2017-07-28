# Model-Free Renewables Scenario Generation Using Generative Adversarial Networks

This repository contains source code necessary to reproduce the results presented in the following paper:

Model-Free Renewables Scenario Generation Using Generative Adversarial Networks
by Yize Chen, Yishen Wang, Daniel Kirschen and Baosen Zhang

This repository can be used for general scenario-generation 

## Motivation
Engineers need an efficient and scalable technique to capture and model the dynamics of time-series scenarios as well as spatio-temporal scenarios for renewables generation process. Traditional model-based methods are proposed with many model assumptions, while on the other side these models are hard to scale into power generation process at different locations.

In this project, we proposed to use the set of generative model, Generative Adversarial Networks, to bring out a data-driven solution for scenario generation problem. 

## Language and Dependencies

We used Python to implement the algorithm. Some data processing work was finished in Matlab. Specifically, we used the open-source Python package [Tensorflow](https://www.tensorflow.org/) to train the neural network models.

To run the code, you need installing the standard packages of numpy, pandas, ipdb and matplotlib. In Linux, you can install these packages via `pip install numpy pandas ipdb matplotlib`.


## Run Experiments with Pre-Processed Datasets

To reproduce the results shown in paper, we also updated the three datasets, which corresponding to wind scenario generation, solar scenario generation and wind spatio-temporal scenario generation. Labels are added to each training sample by given events (e.g., wind'sample's mean value, solar sample's month). All datasets in these paper are produced from NREL wind



## Questions?

Please email [me](http://blogs.uw.edu/yizechen/) at yizechen@uw.edu if you have any questions!
