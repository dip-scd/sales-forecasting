import pandas as pd
from typing import Tuple
import pandas as pd
import numpy as np
from typing import Any, Optional
from sklearn.linear_model import LinearRegression

from src.features.timeseries_augumentation import get_week_phase, get_month_phase


def create_data_for_model(df_data: pd.DataFrame, max_lag: int = 9, 
                          target_shift: int = 1, 
                          timeseries_transformer: Optional[Any] = None) -> pd.DataFrame:
    """
    Create a dataset for time series forecasting model.

    Args:
        df_data (pd.DataFrame): Input data frame containing the time series data.
        max_lag (int, optional): Maximum number of lags to create. Defaults to 9.
        target_shift (int, optional): Number of periods to shift the target variable. Defaults to 1.
        timeseries_transformer (Optional[Any], optional): Transformer to apply to the time series data. Defaults to None.

    Returns:
        pd.DataFrame: Data frame with lag features and target variable.
    """
    sr_x_transformed = df_data['x']
    if timeseries_transformer is not None:
        sr_x_transformed = timeseries_transformer.transform_forward(df_data['x'])
    df_data_transformed = sr_x_transformed.to_frame()

    df_data_m = df_data_transformed.dropna().copy()
    df_data_m.columns.rename(None, inplace=True)
    df_data_m.rename(columns={df_data_m.columns[0]: 'x'}, inplace=True)

    df_x_features = create_lag_features(df_data_m, max_lag)
    df_data_ext = add_target(df_x_features, target_shift)
    return df_data_ext

def create_lag_features(df: pd.DataFrame, max_lag: int = 9) -> pd.DataFrame:
    """
    Create lag features from dataframe column x up to max_lag.

    Args:
        df: Input dataframe 
        max_lag: Max lag to create features for

    Returns:
        DataFrame containing lag features
    """
    dict_x_features = {}
    for num_shift in range(max_lag,-1,-1):
        dict_x_features[('x', f'-{num_shift}')] = df['x'].shift(num_shift)

    df_x_features = pd.DataFrame(dict_x_features)
    return df_x_features


def add_target(df: pd.DataFrame, target_shift: int = 1) -> pd.DataFrame:
    """
    Add target column y to dataframe df by shifting 
    column x by target_shift.

    Args:
        df: Input dataframe
        target_shift: Number of periods to shift x by to create y

    Returns:
        DataFrame with target column y added
    """
    dict_tgt = {}
    dict_tgt[('y', 1)] = df[('x', '-0')].shift(-target_shift)
    df_tgt = pd.DataFrame(dict_tgt)
    df_ext = df.join(df_tgt).dropna()
    return df_ext

def add_baseline_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline column using last value from x column.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with baseline column added
    """
    df = df.copy()
    df[('baseline', 'last')]  = df[('x', '-0')]
    return df


def split_data_train_val_test(df_data: pd.DataFrame, num_val_ratio: float = 0.1, num_test_ratio: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a Pandas DataFrame into train, validation and test subsets.
    
    Args:
        df_data: Source DataFrame. 
        num_val_ratio: Ratio of validation set relative to entire dataset.
        num_test_ratio: Ratio of test set relative to entire dataset.
            If None, test set ratio is equal to validation set ratio.
            
    Returns:
        Tuple of DataFrames for train, validation and test.
    """
    if num_test_ratio is None:
        num_test_ratio = num_val_ratio

    num_train_ratio = 1 - num_test_ratio - num_val_ratio

    train_size = int(len(df_data) * num_train_ratio)
    val_size = int(len(df_data) * num_val_ratio)

    df_train = df_data.iloc[:train_size]
    df_val = df_data.iloc[train_size:train_size+val_size]
    df_test = df_data.iloc[train_size+val_size:]

    return df_train, df_val, df_test


def get_x_y(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Get X and y arrays from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing x and y data
        
    Returns:
        Tuple[pd.Series, pd.Series]: X and y arrays
    """
    X = df['x']
    y = df['y']
    return X, y

def create_baseline_model(df_train: pd.DataFrame) -> LinearRegression:
    """Create a baseline linear regression model on the training data.
    
    Args:
        df_train (pd.DataFrame): Training data
    
    Returns:
        LinearRegression: Fitted baseline model
    """
    baseline_model = LinearRegression()
    X_train, y_train = get_x_y(df_train)
    baseline_model.fit(X_train, y_train)
    return baseline_model

def get_baseline_prediction(df: pd.DataFrame, baseline_model: LinearRegression) -> np.ndarray:
    """Get predictions from baseline model on provided data.
    
    Args:
        df (pd.DataFrame): Data to predict on
        baseline_model (LinearRegression): Fitted baseline model
        
    Returns:
        np.ndarray: Predictions from baseline model
    """
    X, _ = get_x_y(df)
    yhat = baseline_model.predict(X)
    return yhat

def augument_with_linear_baseline(
    df: pd.DataFrame, 
    validation_ratio: float, 
    test_ratio: float
) -> pd.DataFrame:
    """Augment a DataFrame with predictions from a baseline linear model.
    
    Args:
        df (pd.DataFrame): Source data
        validation_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        
    Returns:
        pd.DataFrame: DataFrame augmented with baseline predictions
    """
    df_train, _, _ = split_data_train_val_test(df, validation_ratio, test_ratio)
    baseline_model = create_baseline_model(df_train)
    baseline_yhat_all = get_baseline_prediction(df, baseline_model)
    df[('baseline', 'linear_regression')] = baseline_yhat_all.reshape(-1)
    return df


def augument_with_baseline_predictions(
    df: pd.DataFrame, 
    validation_ratio: float, 
    test_ratio: float,
    is_differenced: bool = True
) -> pd.DataFrame:
    """Augment a DataFrame with baseline predictions.
    
    Args:
        df (pd.DataFrame): Source data
        validation_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        is_differenced (bool): Whether the data is differenced
        
    Returns:
        pd.DataFrame: DataFrame augmented with baseline predictions
    """
    if is_differenced:
        df[('baseline', 'naive')] = 0.
    else:
        df[('baseline', 'naive')] = df[('x', '-0')]

    # adding linear regression baseline prediction
    df = augument_with_linear_baseline(df, validation_ratio, test_ratio)
    return df

def augument_with_periodic_phases(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Augment a DataFrame by adding columns for periodic phases.
    
    Args:
        df (pd.DataFrame): Source data
        
    Returns:
        pd.DataFrame: DataFrame with added periodic phase columns 
    """
    df[('x', 'week_phase')] = pd.Series(df.index).apply(get_week_phase).values
    df[('x','month_phase')] = pd.Series(df.index).apply(get_month_phase).values
    return df

def augument_the_data(
    df: pd.DataFrame, 
    validation_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    do_augument_with_periodic_phases: bool = False
) -> pd.DataFrame:
    """Augment a DataFrame with baseline predictions and periodic phases.
    
    Args:
        df (pd.DataFrame): Source data
        validation_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        do_augument_with_periodic_phases (bool): Whether to add periodic phases
        
    Returns:
        pd.DataFrame: Augmented DataFrame
    """
    # Adding week and month phase to the data in order to help the model to take into account potential periodicities
    # Adding year phase woukd be risky. It would work as an index variable that would help to memoize the predictions
    if do_augument_with_periodic_phases:
        df = augument_with_periodic_phases(df)

    df = augument_with_baseline_predictions(
        df, validation_ratio, 
        test_ratio, is_differenced = True)

    return df

def get_x_y(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Get X and y arrays from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing x and y data
        
    Returns:
        Tuple[pd.Series, pd.Series]: X and y arrays
    """
    X = df['x']
    y = df['y']
    return X, y

def create_baseline_model(df_train: pd.DataFrame) -> LinearRegression:
    """Create a baseline linear regression model on the training data.
    
    Args:
        df_train (pd.DataFrame): Training data
    
    Returns:
        LinearRegression: Fitted baseline model
    """
    baseline_model = LinearRegression()
    X_train, y_train = get_x_y(df_train)
    baseline_model.fit(X_train, y_train)
    return baseline_model

def get_baseline_prediction(df: pd.DataFrame, baseline_model: LinearRegression) -> np.ndarray:
    """Get predictions from baseline model on provided data.
    
    Args:
        df (pd.DataFrame): Data to predict on
        baseline_model (LinearRegression): Fitted baseline model
        
    Returns:
        np.ndarray: Predictions from baseline model
    """
    X, _ = get_x_y(df)
    yhat = baseline_model.predict(X)
    return yhat

def augument_with_linear_baseline(
    df: pd.DataFrame, 
    validation_ratio: float, 
    test_ratio: float
) -> pd.DataFrame:
    """Augment a DataFrame with predictions from a baseline linear model.
    
    Args:
        df (pd.DataFrame): Source data
        validation_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        
    Returns:
        pd.DataFrame: DataFrame augmented with baseline predictions
    """
    df_train, _, _ = split_data_train_val_test(df, validation_ratio, test_ratio)
    baseline_model = create_baseline_model(df_train)
    baseline_yhat_all = get_baseline_prediction(df, baseline_model)
    df[('baseline', 'linear_regression')] = baseline_yhat_all.reshape(-1)
    return df


def augument_with_baseline_predictions(
    df: pd.DataFrame, 
    validation_ratio: float, 
    test_ratio: float,
    is_differenced: bool = True
) -> pd.DataFrame:
    """Augment a DataFrame with baseline predictions.
    
    Args:
        df (pd.DataFrame): Source data
        validation_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        is_differenced (bool): Whether the data is differenced
        
    Returns:
        pd.DataFrame: DataFrame augmented with baseline predictions
    """
    if is_differenced:
        df[('baseline', 'naive')] = 0.
    else:
        df[('baseline', 'naive')] = df[('x', '-0')]

    # adding linear regression baseline prediction
    df = augument_with_linear_baseline(df, validation_ratio, test_ratio)
    return df


def augument_with_periodic_phases(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Augment a DataFrame by adding columns for periodic phases.
    
    Args:
        df (pd.DataFrame): Source data
        
    Returns:
        pd.DataFrame: DataFrame with added periodic phase columns 
    """
    df[('x', 'week_phase')] = pd.Series(df.index).apply(get_week_phase).values
    df[('x','month_phase')] = pd.Series(df.index).apply(get_month_phase).values
    return df


def augument_the_data(
    df: pd.DataFrame, 
    validation_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    do_augument_with_periodic_phases: bool = False,
    is_differenced: bool = True
) -> pd.DataFrame:
    """Augment a DataFrame with baseline predictions and periodic phases.
    
    Args:
        df (pd.DataFrame): Source data
        validation_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        do_augument_with_periodic_phases (bool): Whether to add periodic phases
        
    Returns:
        pd.DataFrame: Augmented DataFrame
    """
    # Adding week and month phase to the data in order to help the model to take into account potential periodicities
    # Adding year phase woukd be risky. It would work as an index variable that would help to memoize the predictions
    if do_augument_with_periodic_phases:
        df = augument_with_periodic_phases(df)

    df = augument_with_baseline_predictions(
        df, validation_ratio, 
        test_ratio, is_differenced = is_differenced)

    return df

def create_augumented_data_for_model(df_data: pd.DataFrame, 
                                    max_lag: int = 9, 
                                    target_shift: int = 1,
                                    validation_ratio: float = 0.1,
                                    test_ratio: float = 0.1,
                                    timeseries_transformer=None,
                                    is_differenced: bool = False,
                                    augument_with_periodic_phases: bool = False,
                                    ) -> pd.DataFrame:
    """
    Create augmented data for a time series forecasting model.

    Args:
        df_data (pd.DataFrame): Input data frame containing the time series data.
        max_lag (int, optional): Maximum lag to consider for creating lagged features. Defaults to 9.
        target_shift (int, optional): Number of periods to shift the target variable. Defaults to 1.
        timeseries_transformer (callable, optional): Transformer function to apply to the time series data. Defaults to None.
        is_differenced (bool, optional): Whether the data has been differenced. Defaults to False.
        augument_with_periodic_phases (bool, optional): Whether to augment the data with periodic phases. Defaults to False.

    Returns:
        pd.DataFrame: Augmented data frame with lagged features and target variable.
    """
    df_data_ret = create_data_for_model(
        df_data, max_lag=max_lag, 
        target_shift=target_shift,
        timeseries_transformer=timeseries_transformer)
    df_data_ret = augument_the_data(
        df_data_ret, validation_ratio, 
        test_ratio, 
        is_differenced = is_differenced,
        do_augument_with_periodic_phases=augument_with_periodic_phases)

    return df_data_ret