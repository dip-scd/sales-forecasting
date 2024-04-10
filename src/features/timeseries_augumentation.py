import datetime
from copy import deepcopy

def get_year_phase(d: datetime) -> float:
    """Given a datetime object, returns a float from 0 to 1 
    representing the phase of the year.
    
    1st Jan 00:00 is 0.0, 31st Dec 23:59:59 is 1.0. 
    Dates in leap years return the same value as non-leap years.
    
    Args:
        d (datetime): Input datetime
        
    Returns:
        float: Phase of the year from 0 to 1
    """
    d = deepcopy(d)
    #forcing year to be leap so that dates in leap and non-leap years result in the same phase value
    d = d.replace(year=2000) 
    year_start = datetime.datetime(d.year, 1, 1)
    year_end = datetime.datetime(d.year, 12, 31, 23, 59, 59)
    
    phase = (d - year_start) / (year_end - year_start)
    return phase

def get_week_phase(d: datetime) -> float:
    """Given a datetime object, returns a float from 0 to 1 
    representing the phase of the week.
    
    Week starts on Monday.
    1st Jan 00:00 is 0.0, Sun 23:59:59 is 1.0.
    
    Args:
        d (datetime): Input datetime 
        
    Returns:
        float: Phase of the week from 0 to 1
    """
    d = deepcopy(d)
    dz = d.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = dz - datetime.timedelta(days=d.weekday())
    week_end = week_start + datetime.timedelta(days=7)

    phase = (d - week_start) / (week_end - week_start)
    return phase

def get_month_phase(d: datetime) -> float:
    """Given a datetime object, returns a float from 0 to 1 
    representing the phase of the month.
    
    1st day of month 00:00 is 0.0, last day of month 23:59:59 is 1.0.
    
    Args:
        d (datetime): Input datetime
        
    Returns:
        float: Phase of the month from 0 to 1
    """
    # returns same value for the same day number in different months
    d = deepcopy(d)
    dz = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_start = dz
    month_end = month_start + datetime.timedelta(days=31)
    phase = (d - month_start) / (month_end - month_start)
    return phase