
import pandas as pd
from IPython.display import display, HTML

from src.analysis.stat_tests import get_dict_gausian_tests, get_dict_unitroot_tests

def table_row_from_dict(dict_str_results: dict[str, str]) -> str:
  """Generate an HTML table row from a dictionary of string results.

  Args:
    dict_str_results: Dictionary with string keys and string values.

  Returns:
    str: HTML table row string.
  """
  
  table = "<tr>"
  for key in dict_str_results:
    table += "<td>" + '<br>'.join(dict_str_results[key].split('\n')) + "</td>"
  table += "</tr>"
  return table

def table_from_lst_dicts(lst_dicts: list[dict[str, str]]) -> str:
  """Generate an HTML table from a list of dictionaries.
  
  Args:
    lst_dicts: List of dictionaries with string keys and values.
  
  Returns:
    str: HTML table string.
  """

  table = "<table>"
  for dict_str_results in lst_dicts:
    table += table_row_from_dict(dict_str_results)
  table += "</table>"
  return table

def show_stat_test_results(sr_x: pd.Series) -> None:
    """Display statistical test results for stationarity.
    
    Args:
        sr_x: Time series data
        
    Returns:
        None
    """
    
    sr_x = sr_x.dropna()
    
    lst_dict_str_results = [
        get_dict_gausian_tests(sr_x),
        get_dict_unitroot_tests(sr_x),
    ]

    display(HTML(table_from_lst_dicts(lst_dict_str_results)))