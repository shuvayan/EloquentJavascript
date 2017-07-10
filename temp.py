# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd 
#import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#matplotlib inline

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#Import libraries for modelling:
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import xgboost

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir = 'C:/Users/shuvayan.das/Documents/AttritionModelling'
attrition = pd.read_csv('C:/Users/shuvayan.das/Documents/AttritionModelling/Attrition.csv')
attrition.head()
#Drop the employee code:
attrition.isnull().any()
#Only department has missing values,assign a seperate category to these records
attrition_df = attrition.fillna("unknown")
attrition_df.isnull().any()

attrition_df.columns.to_series().groupby(attrition_df.dtypes).groups
# The target column is in integer format,change to categorical
attrition_df['Terminated'] = attrition_df['Terminated'].astype('category')

# There are some records where the Tenure is negative or the Tenure is less than LastPromoted Time
if ((attrition_df['Tenure'] <= attrition_df['TimeLastPos']) | (attrition_df['Tenure'] <= 0)):
    attrition_df['Flag_Variable'] = 1 
else:
    attrition_df['Flag_Variable'] = 0

attrition_df.to_csv("Attrition_processed.csv")



#Distribution of the dataset
# Plotting the KDEplots
f, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=False, sharey=False)

# Defining our colormap scheme
s = np.linspace(0, 3, 10)
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

# Generate and plot
x = attrition_df['Age'].values
y = attrition_df['Tenure'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])
axes[0,0].set( title = 'Age against Tenure')

cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
# Generate and plot
x = attrition_df['Age'].values
y = attrition_df['Annual Income'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])
axes[0,1].set( title = 'Age against Annual Income')

cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
# Generate and plot
x = attrition_df['TimeLastPos'].values
y = attrition_df['Age'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])
axes[0,2].set( title = 'TimeLastPos against Age')

cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
# Generate and plot
x = attrition_df['Tenure'].values
y = attrition_df['Last Rating'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])
axes[1,1].set( title = 'Tenure against Last Rating')

cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
# Generate and plot
x = attrition_df['Tenure'].values
y = attrition_df['Annual Income'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])
axes[2,0].set( title = 'Years at company against Annual Income')

f.tight_layout()


# 3D Plots:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = attrition_df['Tenure']
y = attrition_df['TimeLastPos']
z = attrition_df['LastRating']
c = attrition_df['Terminated']
_ = ax.scatter(xs=x, ys=y, zs=z, c=c)
_ = ax.set_xlabel('Tenure')
_ = ax.set_ylabel('Annual Income')
_ = ax.set_zlabel('LastRating')
_ = plt.title('Plot 1: Multivariate Visualization of Attrition by Color(red if left)')
plt.show()


# creating a list of only numerical values for correlation.
numerical = ['Tenure','TimeLastPos','Annual Income','Age','LastRating']
       
       
data = [
    go.Heatmap(
        z= attrition[numerical].astype(float).corr().values, # Generating the Pearson correlation
        x=attrition[numerical].columns.values,
        y=attrition[numerical].columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
        opacity = 1.0
        
    )
]

layout = go.Layout(
    title='Pearson Correlation of numerical features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
    
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')

# Define a dictionary for the target mapping
target_map = {'Yes':1.0, 'No':0.0}
# Use the pandas apply method to numerically encode our attrition target variable
attrition["Attrition_numerical"] = attrition_df["Terminated"].apply(lambda x: target_map[x])

#Pairplot Visualisations

# Refining our list of numerical variables
g = sns.pairplot(attrition[numerical], hue='Attrition_numerical', palette='seismic',
                 diag_kind = 'kde',diag_kws=dict(shade=True),hue = "Terminated")
g.set(xticklabels=[])

