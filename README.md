![logo](https://raw.githubusercontent.com/giotto-ai/giotto-tda/master/doc/images/tda_logo.svg)

# Analysis of acoustic signals with giotto-time

## What is it?
This repository contains the code for the blog post ['Analysis of acoustic signals with giotto-time']() where we use the Python time series library [giotto-time](https://github.com/giotto-ai/giotto-time) to predict the type of acoustic space based on crest factor detrending and sorted density measure. 

The 'deliver_tutorial.ipynb' showcases the applications of features of giotto-time to acoustic time series analysis:
* uses **detrending based on [crest factor](https://en.wikipedia.org/wiki/Crest_factor)** to remove the trend without an assumption on the trend type
* uses **sorted density** measure to characterize within what fraction of peaks is the information hidden

## Getting started
You want to start right away? The easiest way to get started is to create a conda environment as follows:
```
conda create python=3.7 --name time -y
conda activate time
pip install -r requirements.txt
```
Then the notebook 'deliver_tutorial.ipynb' will walk you through the analysis and the prediction steps.

## Data
The data used for this project was collected part of a join project between EPFL, Microsoft and Aalto university. The data comes from the simulated and measured acoustic impulse responses used in the following publication:
H. P. Tukuljac, V. Pulkki, H. Gamper, K. Godin, I. J. Tashev and N. Raghuvanshi, "A Sparsity Measure for Echo Density Growth in General Environments," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 1-5.
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8682878&isnumber=8682151

## Results
The important point of this tutorial is to show that, although rarely treated in similar way, acoustic data and financial data could use similar types of characterization. The sorted density measure presented here can be used for characterization of financial trends, where the emphasis is on detecting rare events rather than relying on pure averaging models, common for time series analysis. 

