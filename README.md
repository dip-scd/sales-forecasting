Brazilian e-commerce company OLIST  
https://www.kaggle.com/datasets/erak1006/brazilian-e-commerce-company-olist  
Sales forecasting.

### Notebooks
* [Raw data exploration and preprocessing](notebooks/1-data_exploration.ipynb) 
* [Exploring the time series properties and forecasting](notebooks/2-forecasting.ipynb)

### Project Organization
------------
    ├── LICENSE
    ├── data
    │   └── raw            <- The original data.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── [1-data_exploration.ipynb](notebooks/1-data_exploration.ipynb) <- Raw data exploration and preprocessing.
    │   │   
    │   └── [2-forecasting.ipynb](notebooks/2-forecasting.ipynb) <- Exploring the time series properties and forecasting.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to process the data
    │   │
    │   ├── features       <- Scripts to turn the processed data into features for modeling
    │   │
    │   ├── visualization  <- Scripts to create visualizations
    │   │
    │   └── utils          <- Additional utils functions
    │
    ├── tests              <- Unit tests
--------

### Running the notebooks
In order to run the notebooks, create a virtual environment based on the requirements.txt file and install the dependencies. Jupyter must be launched from the project root as a working directory.

### Unit tests
```
cd tests
pyhon -m unittest
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
