# Assignment 2 - IN5460

Felix Krause, Johannes Spies, Sina Henning

Mandatory assignment 2 for Artificial Intelligence for Energy Informatics.

[**Assignment**](https://drive.google.com/file/d/17nJ7HWbZPYZQXsJxbWggvyT49YTZW_M-/view) | Deadline 12.11.2023 kl 23:59

![project7](https://github.com/f-krause/uio-AIEI-project-2/assets/93521294/854c632f-28c7-4c5f-8679-34694c5ff4ed)


## Setup
### Environment
Set up virtual environment
```shell	
python -m venv venv
```

Activate custom virtual environment 
```shell
source venv/bin/activate
```

Install required packages
```shell
pip install -r requirements.txt
```

Add environment to Jupyter Notebook
```shell
python -m ipykernel install --user --name=venv
```

### Data
Place **[Dataset.csv:](https://zenodo.org/records/6778401)** in a directory called "data"

Attention! There are two links to datasets in the docs. We are using the one which can be found on zenodo.org

Run the [preprocessing notebook](task0_preprocessing.ipynb) to create the preprocessed datasets used later.


## Run via jupyter notebook
Simply execute the respective notebooks:
* [task1.1_prediction_RNN.ipynb](task1.1_prediction_RNN.ipynb)
* [task1.2_prediction_LSTM.ipynb](task1.2_prediction_LSTM.ipynb)
* [task2.1_2.2_classification_RNN.ipynb](task2.1_2.2_classification_RNN.ipynb)
* [task2.3_classification_LSTM.ipynb](task2.3_classification_LSTM.ipynb)

## Run via shell
Adapt [main.py](src/main.py) to your needs and execute in project root:
```bash
python -m src
```

## Useful tutorials
* **[lstm example](https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=CKEzO1jzKydL)** a jupyter notebook with a time series prediction example using lstm
* **[rnn vs lstm vs gru code](https://www.tertiaryinfotech.com/comparison-of-lstm-gru-and-rnn-on-time-series-forecasting-with-pytorch/)** architectures of the three different models
