# colors_vqa



## Dependencies

### Anaconda
We recommend installing the free Anaconda Python distribution, which includes IPython, Numpy, Scipy, matplotlib, scikit-learn, NLTK, and many other useful packages. This is not required, but it's an easy way to get all these packages installed. Unless you're very comfortable with Python package management and like installing things, this is the option for you!

Please be sure that you download the Python 3 version, which currently installs Python 3.7. The codebase is not compatible with Python 2.

One you have Anaconda installed, it makes sense to create a virtual environment for the course. In a terminal, run

conda create -n nlu python=3.7 anaconda

to create an environment called nlu.

Then, to enter the environment, run

conda activate nlu

To leave it, you can just close the window, or run

conda deactivate

If your version of Anaconda is older than version 4.4 (see conda --version), then replace conda with source in the above (and consider upgrading your Anaconda!).

### PyTorch
The PyTorch library has special installation instructions depending on your computing environment. For Anaconda users, we recommend

conda install pytorch torchvision -c pytorch

For non-Anaconda users, or if you have a CUDA-enabled GPU, we recommend following the instructions posted here:

https://pytorch.org/get-started/locally/

This projects requires at least 1.4.0:

import torch
​
torch.__version__
'1.4.0'

### NLTK data
Anaconda comes with NLTK but not with its data distribution. To install that, open a Python interpreter and run import nltk; nltk.download(). If you decide to download the data to a different directory than the default, then you'll have to set NLTK_DATA in your shell profile. (If that doesn't make sense to you, then we recommend choosing the default download directory!)
