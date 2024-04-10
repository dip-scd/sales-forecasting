# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from datetime import datetime

def load_raw_data(name: str) -> pd.DataFrame:
    """Loads raw data from CSV files.
    
    Args:
      name (str): The name of the CSV file to load (without extension).
    
    Returns: 
      pd.DataFrame: The dataframe loaded from the CSV.
    """
    df = pd.read_csv(f'../data/raw/OLIST/{name}.csv')
    return df

def calc_df_order_products(df_raw_order_items: pd.DataFrame, df_raw_products: pd.DataFrame, df_raw_orders: pd.DataFrame) -> pd.DataFrame:
    """Calculates order products dataframe from raw order items, products and orders dataframes.

    Args:
    df_raw_order_items: Raw order items dataframe
    df_raw_products: Raw products dataframe 
    df_raw_orders: Raw orders dataframe

    Returns:
    pd.DataFrame: Calculated order products dataframe
    """
    df_order_products = df_raw_order_items.merge(
        df_raw_products[['product_id', 'product_category_name']],
        on='product_id', how='left')\
    .merge(
        df_raw_orders[['order_id', 'order_purchase_timestamp', 'order_status']],
        on='order_id', how='left')\
    .drop(
        columns=['order_id', 'order_item_id', 'seller_id', 'shipping_limit_date', 'freight_value'])
    
    df_order_products = df_order_products[
        df_order_products['order_status'].isin(['delivered', 'shipped'])
    ].copy()

    # converting the order_purchase_timestamp to datetime
    df_order_products['order_purchase_timestamp'] = pd.to_datetime(df_order_products['order_purchase_timestamp'])

    # creating a new column with the date of the order
    df_order_products.loc[:,'order_date'] = df_order_products['order_purchase_timestamp'].dt.floor('D')
    return df_order_products


def calc_df_order_products_agg(df_order_products: pd.DataFrame) -> pd.DataFrame:
    """Calculates order products aggregation dataframe from order products dataframe.
    
    Args:
      df_order_products: Raw order products dataframe 
    
    Returns:
      pd.DataFrame: Calculated order products aggregation dataframe with items sold count by product category and date
    """
    df_order_products_agg = df_order_products\
        .groupby(
            ['product_category_name','order_date']
        ).agg(
            {'product_id': 'count'}
        ).reset_index()\
        .rename(
            columns={
                'product_id': 'items_sold',
                'order_date': 'date'
                }
        )
    return df_order_products_agg

def calc_df_product_sales(df_order_products_agg: pd.DataFrame, start_date: datetime, end_date: datetime, num_top_products: int=1) -> pd.DataFrame:
    """Calculates product sales dataframe from order products aggregation dataframe and filters by date range and top selling products.

    Args:
      df_order_products_agg (pd.DataFrame): Order products aggregation dataframe
      start_date (datetime): Start date for filtering
      end_date (datetime): End date for filtering  
      num_top_products (int): Number of top selling products to return 

    Returns:
      pd.DataFrame: Product sales dataframe with items sold count by product category and date filtered by date range and for top selling products
        """

    df_product_sales = df_order_products_agg.pivot(index='date', columns='product_category_name', values='items_sold')
    df_product_sales = df_product_sales.fillna(0)
    df_product_sales = df_product_sales.resample('D').sum()

    df_product_sales_trimmed = df_product_sales[
        (df_product_sales.index >= start_date) &\
        (df_product_sales.index < end_date)
    ]

    df_top_products = df_product_sales_trimmed.sum().sort_values(ascending=False).head(num_top_products)
    df_product_sales_trimmed_top = df_product_sales_trimmed[df_top_products.index]
    df_product_sales_trimmed_top
    return df_product_sales_trimmed_top


def process_data(start_date: datetime, end_date: datetime, num_top_products: int=1) -> pd.DataFrame:
    """Processes raw data from start date to end date for top selling products.
    
    Args:
      start_date (datetime): Start date for filtering  
      end_date (datetime): End date for filtering
      num_top_products (int): Number of top selling products to return
      
    Returns:
      pd.DataFrame: The processed dataset filtered by date range and for top products
    """

    df_raw_order_items = load_raw_data('order_items_dataset')
    df_raw_products = load_raw_data('products_dataset')
    df_raw_orders = load_raw_data('orders_dataset')

    df_order_products = calc_df_order_products(df_raw_order_items, df_raw_products, df_raw_orders)
    df_order_products_agg = calc_df_order_products_agg(df_order_products)
    df_product_sales = calc_df_product_sales(df_order_products_agg, start_date, end_date, num_top_products)
    return df_product_sales


@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    process_data()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
