# colors_vqa

This is the code repository for VQA Color Images (), which is used to develop a model and Visual Question Answering Dataset around Color Images.
Correctly answering the question requires the model to show understanding of the underlying context ie Pragmatics or Grounding. This dataset and model is targeted at Machine Learning Students/Enthusiasts trying to build their intuitions around DeepLearning,VQA concepts, Multi-Task learning and MultiModal Architectures. The ColorImages dataset is a simple but still challenging dataset because the language of color is rich and nuanced. However it is easier to  reason compared to the complexity around models built around VQA datasets from  \cite{agrawal2015vqa} and \cite{DBLP:journals/corr/abs-1902-09506} which are more research oriented. 

## Getting Started
1. Install Software Dependencies listed below. 
2. Clone this repository
3. cd in the folder and download the dataset. (wget https://storage.googleapis.com/vqamodel-mathaix/vqa.tar.gz)
4. Untar (tar -xvzf vqa.tar.gz)
5. Train the model by running "colors_vqa_model.py". It took roughly 8hrs on a Mac to run 10 Ephocs. Running the evaluation also takes time, roughly an hour. 
6. See the following Notebook to get better understanding of the data and how use the Model. (VisualQuestionAnsweringonColors.ipynb)
7. The following file is responsible for generating data color_images.py. 



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
