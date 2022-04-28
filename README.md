# GOODKT

The implementation of the paper *GOODKT: A Dual Graph Online Knowledge Distillation Method for Knowledge Tracin*

## Setup

To run this code you need the following:

    a machine with GPUs
    python3
    numpy, pandas, scipy, scikit-learn and torch packages:
```
pip3 install torch==1.7.0 numpy==1.21.2 pandas==1.4.1 scipy==1.7.3 scikit-learn==1.0.2 tqdm==4.26.3 
```
## Save Log

If you need to save the log, create a `log` folder under the evaluation folder.
There are trained models in the model folder, which can be directly run `KTtest()` in  `run.py`  . 
Of course, you can also train a new models, just  run  `KTtrain()` in `run.py`

## DataSet

In addition to the Assistment2017 data sets in the code, the 09 and ednet data sets mentioned in the paper are in the Google Drive, which you can download with this [link](https://drive.google.com/file/d/1ItqFv0fH6ibTotmflFNeAX0d7PdMaF7B/view?usp=sharing).


## Try using your own dataset!

You can also use your own data set, but note that besides the four row data set, you also need to build the incidence matrix of question and concept of the dataset and put it in the `H` folder.
