# Model-Free Renewables Scenario Generation Using Generative Adversarial Networks

This repository contains source code necessary to reproduce the results presented in the following paper:

[Model-Free Renewables Scenario Generation Using Generative Adversarial Networks](https://arxiv.org/abs/1707.09676)

By Yize Chen, Yishen Wang, Daniel Kirschen and Baosen Zhang, 

Accepted to *IEEE Transaction on Power Systems*, 2018 Special Issue on *Enabling very high penetration renewable energy integration into future power systems* 

The method shown in this repository can be used for general scenario-generation problems in power systems. 

## Motivation
Engineers need an efficient and scalable technique to capture and model the dynamics of time-series scenarios as well as spatio-temporal scenarios for renewables generation process. Traditional model-based methods are proposed with many model assumptions, while on the other side these models are hard to scale into power generation process at different locations.

In this project, we proposed to use the set of generative model, Generative Adversarial Networks, to bring out a data-driven solution for scenario generation problem. 

## Generated Samples
Here we show some generated samples along with samples' autocorrelation
![alt text](https://github.com/chennnnnyize/Renewables_Scenario_Gen_GAN/blob/master/datasets/samples.png)

## Language and Dependencies

We used Python to implement the algorithm. Some data processing work was finished in Matlab. Specifically, we used the open-source Python package [Tensorflow](https://www.tensorflow.org/) to train the neural network models.

To run the code, you need installing the standard packages of numpy, pandas, ipdb and matplotlib. In Linux, you can install these packages via `pip install numpy pandas ipdb matplotlib`. For windows, follow [these instructions](https://www.tensorflow.org/install/install_windows). 


## Run Experiments with Pre-Processed Datasets

Datasets: To reproduce the results shown in paper, we also updated the three datasets, which corresponding to wind scenario generation, solar scenario generation and wind spatio-temporal scenario generation. Labels are added to each training sample by given events (e.g., wind'sample's mean value, solar sample's month). All datasets in these paper are produced and processed from [NREL Wind Integration Datasets](https://www.nrel.gov/grid/wind-integration-data.html) and [NREL Solar Integration Datasets](https://www.nrel.gov/grid/solar-power-data.html). The labels are based on the power generation degrees.

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

To run experiment, run "train.py" in your Python IDE or in command line "python train.py", and select running mode as "temporal", "spatio-temporal" or "event". For temporal and event-based scenario generation, we provide example datasets with 5-minute time resolution, while for spatio-temporal scenario generation, a dataset to generate 24-site 1-day scenarios with 1-hour resolution is applied. Once training completes, it will generate a group of generated scenarios along with the training loss evolution.

## Scenarios Evaluation

To evaluate if generated scenarios are good enough, we suggest two methods to use:

a) Check the loss function, which is the metric we used during training. A converged loss to small value shall indicate good quality of generated scenarios.
![alt text](https://github.com/chennnnnyize/Renewables_Scenario_Gen_GAN/blob/master/datasets/loss.png)

b) Check scenarios' statistical properties, and compare the results with historical records.


## Code References
We thank https://github.com/carpedm20/DCGAN-tensorflow for contributing the initial version of tensorflow convolutional GAN module; https://github.com/igul222/improved_wgan_training for Wasserstein GAN examples.
## Questions?

Please email [me](http://blogs.uw.edu/yizechen/) at yizechen@uw.edu if you have any code or implemenration questions!
