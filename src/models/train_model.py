from typing import Callable, List, Tuple
import lightgbm
from lightgbm import LGBMRegressor
import pandas as pd
from tqdm import tqdm

from src.features.make_model_dataset import get_x_y


def supress_lightgbm_logging():
    class SilentLogger:
        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

    #quick hack to suppress the lightgbm logging that ignores verbose=-1
    lightgbm.basic._LOGGER = SilentLogger()

def enable_lightgbm_logging():
    lightgbm.basic._LOGGER = lightgbm.basic._DummyLogger()

def create_fit_model(df_train: pd.DataFrame, 
                     df_val: pd.DataFrame, *args, **kwargs) -> Tuple[LGBMRegressor, float, float]:
    """
    Create and fit a LightGBM Regressor model on the provided training and validation data.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        *args: Positional arguments to pass to LGBMRegressor.
        **kwargs: Keyword arguments to pass to LGBMRegressor.

    Returns:
        Tuple[LGBMRegressor, float, float]: A tuple containing the fitted model, training error, and validation error.
    """
    
    df_X_train, df_y_train = get_x_y(df_train)
    df_X_val, df_y_val = get_x_y(df_val)

    supress_lightgbm_logging()
    model = LGBMRegressor(*args, **kwargs)
    model.fit(df_X_train, df_y_train)
    enable_lightgbm_logging()

    yhat_train = model.predict(df_X_train)
    yhat_val = model.predict(df_X_val)
    train_error = sklearn.metrics.mean_squared_error(yhat_train, df_y_train)
    val_error = sklearn.metrics.mean_squared_error(yhat_val, df_y_val)
    return model, train_error, val_error

def create_fit_model_cv(df: pd.DataFrame, cv_split: Callable, *args, **kwargs) -> pd.DataFrame:
    """
    Create and fit multiple LightGBM Regressor models using cross-validation splits.

    Args:
        df (pd.DataFrame): Input data.
        cv_split (Callable): A function that generates cross-validation splits from the input data.
        *args: Positional arguments to pass to LGBMRegressor.
        **kwargs: Keyword arguments to pass to LGBMRegressor.

    Returns:
        pd.DataFrame: A DataFrame containing the fitted models, training errors, and validation errors for each cross-validation split.
    """
    lst_models: List[LGBMRegressor] = []
    lst_train_errors: List[float] = []
    lst_val_errors: List[float] = []

    for split in cv_split.split(df):
        df_train_split = df.iloc[split[0]]
        df_val_split = df.iloc[split[1]]
        model, train_error, val_error = create_fit_model(df_train_split, df_val_split, *args, **kwargs)
        lst_models.append(model)
        lst_train_errors.append(train_error)
        lst_val_errors.append(val_error)

    return pd.DataFrame({
        "model": lst_models,
        "train_error": lst_train_errors,
        "val_error": lst_val_errors,
    })

import sklearn
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

def create_fit_models_grid(
        param_grid_definition: dict, 
        df_train: pd.DataFrame, 
        df_val: pd.DataFrame,
        progress = tqdm,
        *args, **kwargs,
        ) -> tuple[list, dict, list]:
    """
    Create and fit multiple LightGBM Regressor models using a grid of hyperparameters.

    Args:
        param_grid_definition (dict): A dictionary defining the grid of hyperparameters to search over.
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        progess (tqdm, optional): A progress bar to display during model training. Defaults to tqdm.
        *args: Positional arguments to pass to LGBMRegressor.
        **kwargs: Keyword arguments to pass to LGBMRegressor.

    Returns:
        tuple[list, dict, list]: A tuple containing:
            - A list of hyperparameter dictionaries used in the grid search.
            - A dictionary mapping hyperparameter tuples to the corresponding model, training error, and validation error.
            - A list of dictionaries containing the model, training error, and validation error for each set of hyperparameters.
    """
    lst_grid_params =  list(ParameterGrid(param_grid_definition))
    lst_grid_models = []

    for params in progress(lst_grid_params):
        model, train_error, val_error = create_fit_model(
            df_train, df_val, 
            *args,
            **kwargs,
            **params
        )

        lst_grid_models.append({
            'model': model, 
            'train_error': train_error,
            'val_error': val_error})  

    print(f'{len(lst_grid_models)} models trained and validated')

    return lst_grid_params, lst_grid_models


def create_fit_models_grid_cv(
        param_grid_definition: dict, 
        df: pd.DataFrame, 
        n_splits: int,
        progress = tqdm,
        *args, **kwargs,
        ) -> tuple[list, dict, list]:
    """
    Create and fit multiple LightGBM Regressor models using a grid of hyperparameters and cross-validation.

    Args:
        param_grid_definition (dict): A dictionary defining the grid of hyperparameters to search over.
        df (pd.DataFrame): Training data.
        n_splits (int): Number of splits for cross-validation.
        progress (tqdm, optional): A progress bar to display during model training. Defaults to tqdm.
        *args: Positional arguments to pass to LGBMRegressor.
        **kwargs: Keyword arguments to pass to LGBMRegressor.

    Returns:
        tuple[list, dict, list]: A tuple containing:
            - A list of hyperparameter dictionaries used in the grid search.
            - A dictionary mapping hyperparameter tuples to the corresponding model, training error, and validation error.
            - A list of dictionaries containing the model, training error, and validation error for each set of hyperparameters.
    """
    lst_grid_params =  list(ParameterGrid(param_grid_definition))
    lst_grid_models = []

    cv_split = TimeSeriesSplit(n_splits=n_splits)

    for params in progress(lst_grid_params):
        df_models = create_fit_model_cv(
            df,
            cv_split,
            *args,
            **kwargs,
            **params
        )

        model = df_models['model'].iloc[-1]

        train_error = df_models['train_error'].mean()
        val_error = df_models['val_error'].mean()

        lst_grid_models.append({
            'model': model, 
            'train_error': train_error,
            'val_error': val_error})  

    print(f'{len(lst_grid_models)} models trained and validated')

    return lst_grid_params, lst_grid_models


def get_df_models(lst_grid_params: list[dict], lst_grid_models: list[dict]) -> pd.DataFrame:
    """
    Creates a DataFrame from lists of grid parameters and grid models.

    Args:
        lst_grid_params (list[dict]): A list of dictionaries containing grid parameters.
        lst_grid_models (list[dict]): A list of dictionaries containing grid models.

    Returns:
        pd.DataFrame: A DataFrame containing the grid parameters and models.
    """
    dict_grid_modelparams = {}
    for p, m in zip(lst_grid_params, lst_grid_models):
        d = {}
        model_key = None
        for k in p.keys():
            d[('param', k)] = p[k]
        for k in m.keys():
            if k != 'model':
                d[('error', k)] = m[k]
            else:
                model_key = m[k]
        dict_grid_modelparams[model_key] = d

    return pd.DataFrame(dict_grid_modelparams).T

