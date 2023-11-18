#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn import linear_model
import requests
from nba_api.stats import endpoints
from matplotlib import pyplot as plt


# In[17]:


# Here I am accessing the leadueleaders module to get acess to season totals

data = endpoints.leagueleaders.LeagueLeaders()

# The "data" variable now has built in functions like getting our dataframe

df = data.league_leaders.get_data_frame()

df



# In[6]:


# Building our regression Model
# Here I use Sci-kit learn to build a linear regression model with two variables FGA and total points

# here we need to divide each variable by games played (GP) to get per game average

x, y = df.FGA/df.GP, df.PTS/df.GP

# our arrays need to be reshaped from 1 dimensional to 2 dimensional

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)



# In[13]:


# Building and fitting our linear regression model

# creating an object that contains the linear models class
#fitting the modeling using FGA (x) and PPG (Y)

model = linear_model.LinearRegression()
model.fit(x,y)

#Getting r2 value and rounding to a two decimal number. 
#getting our predicted y values from x 

r2 = round(model.score(x,y), 2)
predicted_y = model.predict(x)


# In[101]:


# Creating our data visualization using matplotlib

plt.scatter(x, y, s=15, alpha=.5)                            # Scatterplot:  Specfiy size(s) and transparency(alpha) of dots
plt.plot(x, predicted_y, color = 'black')                    # line: Add line for regression line w/ predicted values
plt.title('NBA - Relationship Between FGA and PPG')          # Give it a title
plt.xlabel('FGA per Game')                                   # Label x-axis
plt.ylabel('Points Per Game')                                # Label y-axis
plt.text(10,25, f'R2={r2}') 

#Adding players to the data Viz. League top3 scoring leaders

plt.annotate(df.PLAYER[0],                               #NBAs top scoring player
             (x[0], y[0]),                               # this is our point of annotation
             (x[0]-9, y[0]),                          # These are the coordinate for the text
 
             arrowprops=dict(arrowstyle='-'))          #creating a flat line for the arrow
# joel 
plt.annotate(df.PLAYER[1],
             (x[1],y[1]),
             (x[0]-9, y[0]-4),
             arrowprops=dict(arrowstyle='-'))

plt.annotate(df.PLAYER[2],
             (x[2],y[2]),
             (x[0]-15, y[0]-3),
             arrowprops=dict(arrowstyle='-'))

plt.annotate(f"{df.PLAYER[234]} (median scorer in the league)",
             (x[234],y[234]),
             (x[0]-20.5, y[0]-12),
             arrowprops=dict(arrowstyle='-'),
             fontsize=6.5) 
             
    
            


# In[ ]:


Conclusions

#Players who take a high volume of shots about 17+ start to out perform the models predictions
# Most players are taking less than 7 shots a game
# Also safe to say better players are allowed to shoot more because they are more efficeint than the average

