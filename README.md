# Model-Free Renewables Scenario Generation Using Generative Adversarial Networks

This repository contains source code necessary to reproduce the results presented in the following paper:

Model-Free Renewables Scenario Generation Using Generative Adversarial Networks
by Yize Chen, Yishen Wang, Daniel Kirschen and Baosen Zhang

This repository can be used for general scenario-generation 

## Motivation
Engineers need an efficient and scalable technique to capture and model the dynamics of time-series scenarios as well as spatio-temporal scenarios for renewables generation process. Traditional model-based methods are proposed with many model assumptions, while on the other side these models are hard to scale into power generation process at different locations.

In this project, we proposed to use the set of generative model, Generative Adversarial Networks, to bring out a data-driven solution for scenario generation problem. 

## Generated Samples
![alt text](https://github.com/chennnnnyize/Renewables_Scenario_Gen_GAN/blob/master/datasets/samples.png)

## Language and Dependencies

We used Python to implement the algorithm. Some data processing work was finished in Matlab. Specifically, we used the open-source Python package [Tensorflow](https://www.tensorflow.org/) to train the neural network models.

To run the code, you need installing the standard packages of numpy, pandas, ipdb and matplotlib. In Linux, you can install these packages via `pip install numpy pandas ipdb matplotlib`.


## Run Experiments with Pre-Processed Datasets

Datasets: To reproduce the results shown in paper, we also updated the three datasets, which corresponding to wind scenario generation, solar scenario generation and wind spatio-temporal scenario generation. Labels are added to each training sample by given events (e.g., wind'sample's mean value, solar sample's month). All datasets in these paper are produced from [NREL Wind Integration Datasets](https://www.nrel.gov/grid/wind-integration-data.html) and [NREL Solar Integration Datasets](https://www.nrel.gov/grid/solar-power-data.html).

Run the experiments: 
Please downloads all the .py files in a same folder, and experimental data in a folder "dataset" under the same parent folder.
"model.py" is the function entailing the model structures for GANs networks;
"train.py" is the main function for training and visualization;
"util.py" contatins some helper functions;
"load.py" connects datasets to our model.

We have three modes provided for users:
a) Temporal Scenario Generations; 

b) Spatio-Temporal Scenario Generations;

c) Event-based Scenario Generations;

To run experiment, run "train.py" in your Python IDE or in command line "python train.py", and type in the model as "temporal", "spatio-temporal" or "event". The training code will automatically the corresponding datasets provided in our example and train the model. After training comlete, it will generate a group of generated scenarios along with the training loss evolution.

## Questions?

Please email [me](http://blogs.uw.edu/yizechen/) at yizechen@uw.edu if you have any questions!
