This repository contains code for the paper "Model-Free Renewables Scenario Generation Using Generative Adversarial Networks"

## Convergent Learning: Do different neural networks learn the same representations?

This repository contains source code necessary to reproduce the results presented in the following paper:

```
@inproceedings{li_2016_ICLR
  title={Convergent Learning: Do different neural networks learn the same representations?},
  author={Li, Yixuan and Yosinski, Jason and Clune, Jeff and Lipson, Hod and Hopcroft, John},
  booktitle={International Conference on Learning Representation (ICLR '16)},
  year={2016}
}
```

## Assemble prerequisites

 We used [Caffe](http://caffe.berkeleyvision.org/) to train the models, and computed necessary statistics including pair-wise unit correlations, unit activation mean, pair-wise unit mutual information etc. In this demo, to minimize the effort for you to try out the fun experiments, we have provided a link for you to download all the necessities (pre-trained models, unit statistics, unit visualizations, pre-trained sparse prediction models). 

To run through the demo, you only need the standard packages of IPython, numpy, [networkx](http://networkx.github.io), sklearn and matplotlib packages. Depending on your setup, it may be possible to install these via `pip install ipython numpy matplotlib networkx sklearn`.


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

Please drop [me](http://www.cs.cornell.edu/~yli) a line if you have any questions!
