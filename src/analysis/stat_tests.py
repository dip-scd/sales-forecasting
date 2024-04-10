
import pandas as pd
import numpy as np

from typing import Tuple, Dict, Callable

from arch.unitroot import ADF
from arch.unitroot import KPSS
from arch.unitroot import ZivotAndrews

from scipy.stats import shapiro
from scipy.stats import normaltest
from statsmodels.stats.diagnostic import lilliefors

def get_dict_unitroot_tests(sr_x: pd.Series) -> dict[str, str]:
    """Get dictionary of unit root test results on a time series.

    Args:
        sr_x: Time series to test.

    Returns:
        dict[str, str]: Dictionary with test name as key and result summary as value.
    """
    
    dict_str_results = {}
    for func_test in [ADF, KPSS, ZivotAndrews]:
        test_res = ''
        try:
            test_res = func_test(sr_x).summary().as_text()
        except Exception as e:
            test_res = str(e)
        
        dict_str_results[func_test.__name__] = test_res
    
    return dict_str_results


def get_stat_test_result(test_func: Callable, x: np.ndarray) -> Tuple[float, float, bool]:
    """Apply statistical test on data and return test statistic, p-value, and boolean if p > 0.05.

    Args:
        test_func: Statistical test function to apply.
        x: Data to apply test on.

    Returns:
        Tuple[float, float, bool]: Test statistic, p-value, and boolean if p > 0.05.
    """
    stat, p = test_func(x)
    return (stat, p, p > 0.05)


def calc_dict_gaussian_test_results(sr_x: pd.Series) -> Dict[str, Tuple[float, float, bool]]:
    """Calculate dictionary of Gaussian test results on a time series.
    
    Args:
        sr_x: Time series to test.
        
    Returns:
        Dict[str, Tuple[float, float, bool]]: Dictionary with test name as key 
            and tuple of test statistic, p-value, and boolean if p > 0.05 as value.
    """
    
    dict_gaussian_test_results ={
        e[0]: get_stat_test_result(e[1], sr_x) for e in [
            ('Shapiro-Wilk', shapiro),
            ('D’Agostino’s K^2', normaltest),
            ('Lilliefors', lilliefors)
        ]
    }

    return dict_gaussian_test_results


def get_gausian_test_summary(
    dict_gaussian_test_results: Dict[str, Tuple[float, float, bool]], 
    k: str
) -> str:
    """Generate summary string for a Gaussian test result.

    Args:
        dict_gaussian_test_results: Dictionary of test name to tuple of 
            test statistic, p-value, and boolean if p > 0.05.
        k: Key for the test result to summarize.

    Returns:
        str: Multi-line string summarizing the test result.
    """
    stat, p, passed = dict_gaussian_test_results[k]
    str_passed = 'Gaussian' if passed else 'Not Gaussian'
    lst_str = [
        f'{k} Results',
        '=====================================', 
        f'Test Statistic: {stat}',
        f'P-value: {p}',
        '-------------------------------------',
        f'Conclusion: {str_passed}',
    ]
    str_summary = '\n'.join(lst_str)
    return str_summary


def get_dict_gausian_tests(sr_x: pd.Series) -> Dict[str, str]:
    """Generate dictionary of summary strings for Gaussian tests.
    
    Args:
        sr_x: Time series to test.
        
    Returns:
       Dict[str, str]: Dictionary with test name as key and summary
           string as value.
    """
    
    dict_gaussian_test_results = calc_dict_gaussian_test_results(sr_x)

    dict_str_results = {
        k: get_gausian_test_summary(dict_gaussian_test_results, k) for k in dict_gaussian_test_results.keys()
    }

    return dict_str_results