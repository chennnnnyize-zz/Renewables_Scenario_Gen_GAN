# Model-Free Renewables Scenario Generation Using Generative Adversarial Networks

This repository contains source code necessary to reproduce the results presented in the following paper:

Model-Free Renewables Scenario Generation Using Generative Adversarial Networks
by Yize Chen, Yishen Wang, Daniel Kirschen and Baosen Zhang

This repository can be used for general scenario-generation 

## Motivation
Engineers need an efficient and scalable technique to capture and model the dynamics of time-series scenarios as well as spatio-temporal scenarios for renewables generation process. Traditional model-based methods are proposed with many model assumptions, while on the other side these models are hard to scale into power generation process at different locations.

In this project, we proposed to use the set of generative model, Generative Adversarial Networks, to bring out a data-driven solution for scenario generation problem. 

## Language and Dependencies

We used [Tensorflow](https://www.tensorflow.org/) to train the models

To run through the demo, you need installing the standard packages of IPython, numpy, sklearn and matplotlib packages. Depending on your setup, it may be possible to install these via `pip install ipython numpy matplotlib networkx sklearn`.


## Run experiments with pre-computed statistics

Once the data folder is downloaded, the results can be reproduced using the included IPython notebook `experiments/convergent_learning_notebook.ipynb`.
Start the IPython Notebook server:

```
$ cd experiments
$ ipython notebook
```

Select the `convergent_learning_notebook.ipynb` notebook and execute the included
code. 

_Shortcut: to skip all the work and just see the results, take a look at [this notebook with cached plots](http://nbviewer.jupyter.org/github/yixuanli/convergent_learning/blob/master/experiments/convergent_learning_notebook.ipynb)._



## Questions?

Please email [me](http://blogs.uw.edu/yizechen/) at yizechen@uw.edu if you have any questions!
