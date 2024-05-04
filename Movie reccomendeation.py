#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install -q numpy pandas matplotlib plotly wordcloud scikit-learn')


# In[2]:


import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import save_npz


# In[3]:


data=pd.read_csv("Movie reccomendation.csv")
data


# In[4]:


data.isnull().sum()


# In[5]:


data.fillna('', inplace=True)


# In[6]:


movie_count = data['release_year'].value_counts().sort_index()
fig = go.Figure(data=go.Bar(x=movie_count.index, y=movie_count.values))
fig.update_layout(
    plot_bgcolor='rgb(10,10,10)',
    paper_bgcolor='rgb(17, 17, 17)',  
    font_color='white', 
    title='Number of Movies Released Each Year',  
    xaxis=dict(title='Year'),  
    yaxis=dict(title='Number of Movies')
)
fig.update_traces(marker_color='yellow')
fig.show()


# In[7]:


top_countries = data['country'].value_counts().head(10)
fig = px.treemap(names=top_countries.index, parents=["" for _ in top_countries.index], values=top_countries.values)

fig.update_layout(  
    paper_bgcolor='rgb(102, 35, 35)', 
    font_color='yellow', 
    font_size= 20,
    title='Countries with Highest Number of Movies',
)
fig.show()


# In[8]:


movietype_count = data['type'].value_counts()
fig = go.Figure(data=go.Pie(labels=movietype_count.index, values=movietype_count.values))

fig.update_layout(
    paper_bgcolor='rgb(17, 17, 17)', 
    font_color='Black',  
    title='Distribution of C. Types',
)
fig.update_traces(marker=dict(colors=['yellow']))
fig.show()


# In[9]:


country_movie_counts = data['country'].value_counts()

data = pd.DataFrame({'Country': country_movie_counts.index, 'Movie Count': country_movie_counts.values})

fig = px.choropleth(data_frame=data, locations='Country', locationmode='country names',
                    color='Movie Count', title='Number of Movies Released By Country',
                    color_continuous_scale='Reds', range_color=(0, data['Movie Count'].max()),
                    labels={'Movie Count': 'Number of Movies'})

fig.update_layout(
    plot_bgcolor='rgb(18, 18, 17)',  
    paper_bgcolor='rgb(17, 17, 17)', 
    font_color='yellow' 
)
fig.show()


# In[10]:


ratings       = list(data["duration"].value_counts().index)
rating_counts = list(data["duration"].value_counts().values)

fig = go.Figure(data=[go.Bar(
    x=ratings,
    y=rating_counts,
    marker_color='#E50914'
)])

fig.update_layout(
    title='Movie Durations Distribution',
    xaxis_title='Rating',
    yaxis_title='Count',
    plot_bgcolor='rgba(10, 10, 10, 10)',
    paper_bgcolor='rgba(0, 0, 0, 0.7)',
    font=dict(
        color='white'
    )
)

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




