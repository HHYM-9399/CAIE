import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import PercentFormatter
import geopandas
from geodatasets import get_path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import re
from matplotlib.patches import Patch



def OrderReview_Clean(df):
    df.drop(columns=['review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp'], inplace=True)
    df.drop_duplicates(subset='order_id', keep='last', inplace=True)
    
def Order_Clean(df):
    df.drop(columns=['order_approved_at','order_delivered_carrier_date','order_delivered_customer_date'], inplace = True)
    df.loc[df['order_status'] != 'delivered', 'order_status'] = 'not_delivered'

def Product_Clean(df, df_c):
    df.fillna({'product_category_name': 'unknown'}, inplace=True)
    df.fillna(0, inplace=True)
    df_new = pd.merge(df, df_c, on='product_category_name', how='left')
    df_new.loc[df_new['product_category_name_english'].isnull(), 'product_category_name_english'] = df_new.loc[df_new['product_category_name_english'].isnull(), 'product_category_name']
    df_new.drop(columns=['product_category_name'], inplace=True)
    return df_new

def Product_Bin(df):
    df['product_category_name_english'] = df['product_category_name_english'].str.replace(' ', '_')
    # Define broader mapping categories
    broader_mapping = {
        'home_furnishing': [
            'bed_bath_table', 'furniture_decor', 'housewares', 'home_appliances',
            'furniture_living_room', 'furniture_bedroom', 'furniture_mattress_and_upholstery',
            'home_confort', 'home_construction', 'kitchen_dining_laundry_garden_furniture',
            'home_appliances_2', 'office_furniture', 'portable_kitchen_food_preparers', 'air_conditioning'
        ],
        'electronics_accessories': [
            'computers_accessories', 'telephony', 'electronics', 'small_appliances',
            'fixed_telephony', 'tablets_printing_image', 'pc_gamer'
        ],
        'fashion_accessories': [
            'watches_gifts', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_male_clothing',
            'fashion_underwear_beach', 'fashion_sport', 'fashio_female_clothing', 'fashion_childrens_clothes',
            'luggage_accessories'
        ],
        'health_beauty': [
            'health_beauty', 'perfumery', 'diapers_and_hygiene',
        ],
        'toys_entertainment': [
            'toys', 'cool_stuff', 'consoles_games', 'musical_instruments', 'party_supplies',
        ],
        'sports_pets_outdoors': [
            'sports_leisure', 'garden_tools', 'agro_industry_and_commerce', 'pet_shop'
        ],
        'books_media_stationery': [
            'books_general_interest', 'books_technical', 'books_imported', 'cds_dvds_musicals',
            'dvds_blu_ray', 'music', 'stationery'
        ],
        'auto_tools': [
            'auto', 'construction_tools_construction', 'construction_tools_safety',
            'costruction_tools_garden', 'costruction_tools_tools', 'construction_tools_lights'
        ],
        'fnb': [
            'food_drink', 'food', 'drinks','la_cuisine'
        ],
        'miscellaneous': [
            'unknown', 'market_place', 'signaling_and_security', 'industry_commerce_and_business',
            'christmas_supplies', 'audio', 'art', 'cine_photo', 'arts_and_craftmanship', 'flowers',
            'security_and_services'
        ],
    }

    def map_to_broad_category(product_category_name):
        for category, products in broader_mapping.items():
            if product_category_name in products:
                return category
        return 'miscellaneous'  
    df['category_grouping'] = df['product_category_name_english'].apply(map_to_broad_category)

def show_percentage(g,feature):
    total = len(feature)
    for p in g.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_y() + p.get_height()
        g.annotate(percentage, (x, y), size = 12,ha='center')

def OrderPayment_Clean(df):
    df_new = df.groupby('order_id').agg(
        total_value=('payment_value', 'sum'),
        vouchers_used=('payment_value', 'size'),
        payment_type=('payment_type', lambda x: x[x != 'voucher'].iloc[0] if (x != 'voucher').any() else 'voucher')).reset_index()
    df_new.loc[df['payment_type'] != 'voucher', 'vouchers_used'] = 0
    return df_new

def IQR(OrderPayment):
    Q1 = OrderPayment['total_value'].quantile(0.25)
    Q3 = OrderPayment['total_value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    OrderPayment = OrderPayment[(OrderPayment['total_value'] >= lower_bound) & (OrderPayment['total_value'] <= upper_bound)]
    return OrderPayment

def Merge_all(Orders, OrderItem, Products, OrderPayment, OrderReview, CustomerDataset, Sellers):
    df_total = pd.merge(Orders, OrderItem, on='order_id', how='left')
    df_total = pd.merge(df_total, Products, on='product_id', how='inner')
    df_total = pd.merge(df_total, OrderPayment, on='order_id', how = 'left')
    df_total = pd.merge(df_total, OrderReview, on='order_id', how='left')
    df_total = pd.merge(df_total, CustomerDataset, on='customer_id', how='right')
    df_total = pd.merge(df_total, Sellers, on= 'seller_id', how='left')
    df_total.drop(columns=['review_id','customer_id','shipping_limit_date'])
    return df_total

def NullTester(df_name, df):
    x = df.isnull().sum().sum()
    print(f"For Dataset '{df_name}' there are {x} Null values")

def DupeTester(df_name, df):
    x = df.duplicated().sum()
    print(f"For Dataset '{df_name}' there are {x} Duplicated rows")

def Merging_Clean1(df_total):
    df_total.dropna(subset='order_id', inplace=True)
    df_total.drop(columns='review_id', inplace=True)
    df_total.fillna({'review_score': 0}, inplace=True)

def Merging_Clean2(df_total):
    df_total.dropna(inplace=True)

def DDate(df_total):
    df_total["order_purchase_timestamp"] = pd.to_datetime(df_total["order_purchase_timestamp"])
    df_total["order_estimated_delivery_date"] = pd.to_datetime(df_total["order_estimated_delivery_date"])
    df_total["estimated_delivery_time"] = (df_total["order_estimated_delivery_date"] - df_total["order_purchase_timestamp"]).dt.days

def RepeatBuyer(df_total):
    repeat_buyer = df_total.groupby('customer_unique_id').agg(
        no_of_total_orders=('order_id', 'nunique')
    ).reset_index()
    df_total1 = pd.merge(df_total,repeat_buyer, on='customer_unique_id', how='left')
    return df_total1

def dates(df_total):
    df_total['order_year'] = df_total['order_purchase_timestamp'].dt.year
    df_total['order_month'] = df_total['order_purchase_timestamp'].dt.month

def RepeatBuyer_bin(df_total):
    df_total['repeat_buyer'] = np.where(df_total['no_of_total_orders'] > 1, 1, np.where(df_total['no_of_total_orders'] == 1, 0, -1))
    df_total.drop(columns='no_of_total_orders', inplace=True)

    