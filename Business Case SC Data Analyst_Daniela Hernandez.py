#!/usr/bin/env python
# coding: utf-8

# # Business Case Pepsico

# ## Context
# You have been assigned to support a new area of Business Reporting. As a team, we are constantly asked
# to process new information and creating tools for our clients that will help them make better decisions for
# the company. Accuracy and timeliness are critical, and you have been told that errors will quickly cause the
# business to lose confidence in the new team. In the long-term, your objective is to simplify and find more
# efficient ways to deliver reports to the business. You recognize that the current reports are very manual
# and do not provide many insights to the data.

# ### Import libraries
# 
# To analyze and work with the files, we need to import some libreries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


# In[2]:


app_review = pd.read_csv('googleplaystore_user_reviews.csv')
app_info = pd.read_csv('googleplaystore.csv')


# ### Analizing the first dataset 

# In[3]:


app_review.head()


# The file "Google play store user reviews" contains the following information:
# 
# App: The name of the app
# 
# Translated_Review : The review from the user
# 
# Sentiment : The sentiment analysis from the user, we have 3, Positive, Negative or Neutral.
# 
# Sentiment_Polarity: The value from the sentiment, in this case, 1 is for positive review and 0 is a negative review.
# 
# Sentiment_Subjectivity: The subjectivity from the review, if it is 0 is personal opinion and if is 1 is more clear review. 

# In[4]:


app_review.info()


# In[5]:


app_review.isna().sum()


# We have 70% of the data with NA values. So, I analize the data with the NA values, so we can have more accuracy with the information.

# ### Analizing the second dataset

# In[6]:


app_info


# In[7]:


app_info.info()


# As we can see, we need to convert the data type

# In[8]:


app_info['Reviews'].unique()


# In[9]:


app_info['Reviews'] = pd.to_numeric(app_info['Reviews'],errors='coerce')


# With to_numeric we can convert to a numeric value and also, if it can not convert to a intiger value it will replace it with NaN values.

# In[10]:


app_info['Size'].unique()


# As we can see, we have numbers with a prefix, so we need to eliminate the prefix and keep the number

# In[11]:


app_info['Size_Prefix']=app_info['Size'].str.extract(pat='([a-zA-Z]+)')


# In[12]:


app_info['Size_Prefix']


# In[13]:


app_info['Size']=app_info['Size'].str.extract(pat='(\d{1}[\.\,]?\d+)')


# In[14]:


app_info['Size']


# In[15]:


app_info['Size']=pd.to_numeric(app_info['Size'],errors='coerce')


# In[16]:


app_info.info()


# In[17]:


app_info['Price'].unique()


# In[18]:


app_info['Price']=app_info['Price'].str.extract(pat='(\d+(?:\.\d+)?)')


# In[19]:


app_info['Price']=pd.to_numeric(app_info['Price'], errors='coerce')


# In[20]:


app_info['Price'].unique()


# In[21]:


app_info['Category'] = app_info['Category'].str.lower().str.title()


# In[22]:


app_info['Installs']=app_info['Installs'].str.extract(pat='(\d{1}[\.\,]?\d+)')
app_info['Installs']=app_info['Installs'].str.replace(",","")
app_info['Installs']=app_info['Installs'].astype(float)


# In[23]:


columns = list(app_info.columns)


# In[24]:


new_order = ['App','Category','Rating','Reviews','Size','Size_Prefix','Installs','Type','Price','Content Rating','Genres','Last Updated','Current Ver','Android Ver']
app_info =app_info[new_order]


# With the dataset clean, we can look for the information we need.

# In[25]:


app_info


# In[26]:


app_info.info()


# In[27]:


app_review


# In[28]:


app_review.info()


# We can merge the data using the primary key "App"

# In[29]:


app_total = pd.merge(app_info, app_review, on="App", how='outer')


# In[30]:


app_total


# ## Top five app categories.

# In[31]:


top_five_cat = app_info.groupby('Category')[['Installs']].sum()
top_five_cat = top_five_cat.reset_index()
top_five_cat = top_five_cat.sort_values('Installs', ascending=False)
top_five_cat = top_five_cat.head(5)
top_five_cat


# In[32]:


sns.set_style('darkgrid')
plt.figure(figsize=(15, 8))
sns.barplot(x='Category', y='Installs',data=top_five_cat)
plt.title('Category')
plt.xticks(rotation=80)
plt.ylabel('Installs')
plt.show()


# The category with more installs is Family

# ## Top five rated apps

# In[33]:


top_five_rat = app_info.groupby('Category')[['Rating']].sum()
top_five_rat = top_five_rat.reset_index()
top_five_rat = top_five_rat.sort_values('Rating', ascending=False)
top_five_rat = top_five_rat.head(5)
top_five_rat


# In[34]:


sns.set_style('darkgrid')
plt.figure(figsize=(15, 8))
sns.barplot(x='Category', y='Rating',data=top_five_rat)
plt.title('Category')
plt.xticks(rotation=80)
plt.ylabel('Rating')
plt.show()


# We can see that Family is also the category with the best rating, as well as Game and Tools. 

# ## Which app has more reviews?

# In[42]:


top_five_rv = app_info.groupby('App')[['Reviews']].count()
top_five_rv= top_five_rv.reset_index()
top_five_rv = top_five_rv.sort_values('Reviews', ascending=False)
top_five_rv = top_five_rv.head(5)
top_five_rv


# In[44]:


sns.set_style('darkgrid')
plt.figure(figsize=(15, 8))
sns.barplot(x='App', y='Reviews',data=top_five_rv)
plt.title('App')
plt.xticks(rotation=80)
plt.ylabel('Reviews')
plt.show()


# Again, the category "Family" has more reviews than the others.

# ## Which app is the least liked by the users?

# In[37]:


app_review


# In[38]:


app_least = app_review.query("Sentiment == 'Negative'")
app_least = app_least.groupby('App')['Sentiment'].count()
app_least = app_least.reset_index()
app_least = app_least.sort_values('Sentiment', ascending = False)
app_least=app_least.head(5)


# In[39]:


sns.set_style('darkgrid')
plt.figure(figsize=(15, 8))
sns.barplot(x='App', y='Sentiment',data=app_least)
plt.title('App')
plt.xticks(rotation=80)
plt.ylabel('Sentiment')
plt.show()


# If we choose to find the least app liked by the user based on the column Sentiment, the app "Angry Birds Classic" is the winner, with more Negative Sentiment. 

# With the data clean, now we can extract the information so we can work with the visualization with PowerBI.

# In[40]:


app_info.to_csv('googleplaystore2.csv')


# In[41]:


app_review.to_csv('googleplaystore_user_reviews2.csv')


# # Conclusion

# 
# As a conclusion, and in a general manner, the analysis process carried out for the provided dataset will be explained. The ETL (extract, transform, load) process was employed to analyze the data. 
# 
# Firstly, in Python, the libraries to be used are imported, and for the initial process, the Pandas library is utilized, which helps extract information from a file, in this case, from files in CSV format. 
# 
# Once the information is extracted, we proceed to transform or clean the data. In the case of the "app_review" dataset, there are 6 columns and 64,295 rows. 
# 
# The columns are as follows: 
# App (providing the application name), Translated Review (the user's review), Sentiment (indicating if the review is positive, negative, or neutral), Sentiment Polarity (with a range where 1 means a positive review and 0 means a negative review), and Sentiment Subjectivity (indicating the level of objectivity in the review, with 0 being a personal review and 1 indicating a review with solid grounds). 
# 
# The cleaning process is as follows: We checked for NA data, revealing 26,868 results for the columns "Translated Review," "Sentiment," "Sentiment Polarity," and "Sentiment Subjectivity," indicating an approximate 70% of NA values. To avoid biasing the information, the NA values were left as they are.
# 
# For the second dataset "app_info," there are 13 columns and 10,841 rows. 
# 
# The columns include App (providing the application name), Category (indicating the application's category), Rating (the application's rating), Reviews (the application's reviews), Size (the application's size), Installs (the number of downloads for the application), Content Rating (the target audience for the application), Genres (the application's genre), Last updated (the application's last update), Current Version (the application's version), and Android Version (the Android version required). 
# 
# The cleaning of this dataset involved changing the data type for Reviews, Rating, Size, Installs, and Price, as well as removing prefixes that could hinder proper data usage.
# 
# Once the data is cleaned, questions posed are addressed:
# 
# What are the top 5 categories?
# Family is found to be the category with the highest downloads.
# 
# What are the top 5 categories with the best ratings?
# Similarly, Family is identified as the category with the highest ratings within the Google Play Store.
# 
# Which application has the most reviews?
# Roblox is determined to be the application with the most reviews.
# 
# Which application has the most negative reviews?
# Angry Birds emerges as the application with the most negative reviews.
# 
# Having completed the cleaning of the dataset, the data is extracted. This concludes the extraction, transformation, and cleaning of the data.
# 

# In[ ]:




