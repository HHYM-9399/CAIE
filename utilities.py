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

def OrderReview(df):
    df.drop(columns=['review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp'], inplace=True)
    