import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
import plotly.graph_objs as go
import plotly.express as px
import io
from base64 import b64encode


# Function to create dummy variables
def create_dummy_df(df, cat_cols, dummy_na=False):
	'''
	INPUT:
	df - pandas dataframe with categorical variables you want to dummy
	cat_cols - list of strings that are associated with names of the categorical columns
	dummy_na - Bool whether you want to dummy NA values or not

	OUTPUT:
    df - new dataframe with following characteristics:
        1. contains all columns that were not specified as categorical
        2. removes all the original columns in cat_cols
        3. dummy columns for each of the categorical columns in cat_cols
        4. use a prefix of the column name with an underscore (_) for separating
        5. if dummy_na is True - it also contains dummy columns for NaN values
    '''
	for col in cat_cols:
		try:
			df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
		except:
			continue
	return df

def cleandata(dataset):
	"""Clean my friend's biking data

    Keeps data range of dates in keep_columns variable and data for the top 10 economies
    Reorients the columns into a year, country and value
    Saves the results to a csv file

    Args:
        dataset (str): name of the csv data file

    Returns:
        None

	"""    
	df = pd.read_csv(dataset)

	# Discard all columns with 75%+ null values
	df = df[list(df.columns[df.isnull().sum() <= 0.25*len(df)])]

	# Discard additional columns that have no meaning or are redundant
	discard = ['Activity Name', 'Activity ID', 'Commute', 'Filename', 'Commute.1','Distance', 'Elapsed Time.1', 'Bike']
	df = df.drop(discard, axis = 1)

    # output clean csv file
	return df


def data_wrangle(df):
	"""Clean my friend's biking data further
	1. Datetime values
	2. Categorical values
	3. Null values

    Process the data ready for machine learning models.
	Predicting column 'Activity Gear'

    Args:
        df (dataframe) : dataframe to wrangle

    Returns:
        df (dataframe) : wrangled dataframe

    """   
	# Categorical columns
	# Fix Date column into Year, Month, Day, Hour
	time = df['Activity Date'].astype('datetime64[ns]')
	yr,mon,d,h = [],[],[],[]
	for i in time:
		yr.append(i.year)
		mon.append(i.month)
		d.append(i.day)
		h.append(i.hour)
	df['Year'] = yr
	df['Month'] = mon
	df['Day'] = d
	df['Hour'] = h
	df = df.drop(['Activity Date'], axis=1) # drop original

	# Create dummy variable
	df = create_dummy_df(df, ['Activity Type'], dummy_na = True)

	# Imputation functions
	fill_mean = lambda col: col.fillna(col.mean()) # function for imputating mean
	fill_median = lambda col: col.fillna(col.median()) # function for imputating median

	# impuation on mean
	fill_df = df[['Elevation Gain', 'Average Watts', 'Calories', 'Max Speed', 'Max Grade']].apply(fill_mean, axis=0) 
	fill_df = pd.concat([fill_df, df.drop(['Elevation Gain', 'Average Watts', 'Calories', 'Max Speed', 'Max Grade'], axis=1)], axis=1)
	# imputation on median
	fill_df_med = df[['Athlete Weight', 'Bike Weight', 'Elevation Low', 'Elevation High']].apply(fill_median, axis=0)
	filled_df = pd.concat([fill_df.drop(['Athlete Weight', 'Bike Weight', 'Elevation Low', 'Elevation High'], axis = 1), fill_df_med], axis=1)
	# Alternative solution to null values by dropping all
	dropped_df = df.dropna()

	# Drop null values for 'Activity Gear' 
	# Note: alternative solution is to change na values -> no bike(on foot)
	filled_df = filled_df.dropna()

    # output clean csv file
	return filled_df

def return_figures():
	"""Creates four plotly visualizations

    Args:

        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """
	# first chart is histogram plot
	graph_one = []

	df = cleandata('data/activities.csv')
	filled_df = data_wrangle(df)
	X = filled_df['Distance.1'].values.reshape(-1, 1)
	
	model = LinearRegression()
	model.fit(X, filled_df['Calories'])

	x_range = np.linspace(X.min(), X.max(), 100)
	y_range = model.predict(x_range.reshape(-1, 1))

	graph_one.append(
		px.scatter(
			filled_df, 
			x='Distance.1', 
			y='Calories', opacity=0.65
		).add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
	)
	layout_one = dict(title = 'Calories over Distance', xaxis = dict(title = 'Distance'),yaxis = dict(title = 'Calories'))

	# second chart
	df_two = cleandata('data/activities.csv')
	graph_two = []
	graph_two.append(
		go.Bar(
			x=df_two.Calories,
			y=df_two['Moving Time']
		)
	)

	
	layout_two = dict(title = 'Testing')

	# append all charts to the figures list
	figures = []
	figures.append(dict(data=graph_one[0], layout=layout_one))
	figures.append(dict(data=graph_two, layout=layout_two))
	
	
	return figures