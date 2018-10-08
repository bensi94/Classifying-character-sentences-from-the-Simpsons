#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports and read the simpson data in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[2]:


simpsons_df = pd.read_csv('simpsons_script_lines.csv', error_bad_lines=False, warn_bad_lines=False, low_memory=False, index_col='id')

def dict_from_csv(csv_file_name):
    with open(csv_file_name, mode='r') as infile:
        next(infile)
        reader = csv.reader(infile)
        return_dict = {int(rows[0]):rows[1] for rows in reader}
        return return_dict

character_dict = dict_from_csv('simpsons_characters.csv')
location_dict = dict_from_csv('simpsons_locations.csv') 


# In[3]:


# Clear up the data
# We need to have value in character_id, location_id, and normalized_text so we drop nan rows
simpsons_df = simpsons_df.dropna(subset=['character_id', 'location_id', 'normalized_text'])
# In number, timestamp_in_ms, raw_text, and speaking_line we have no extra information so we drop those columns
simpsons_df = simpsons_df.drop(['number', 'timestamp_in_ms', 'raw_text', 'speaking_line'], axis=1)
# Change numeric values to numeric
simpsons_df['character_id'] = pd.to_numeric(simpsons_df['character_id'])
simpsons_df['location_id'] = pd.to_numeric(simpsons_df['location_id'], downcast='integer')
simpsons_df['word_count'] = pd.to_numeric(simpsons_df['word_count'], downcast='integer')


# In[4]:


# Count how many lines each character has
character_line_counts = Counter(simpsons_df['character_id']).most_common()

most_common_characters = []

# If character has fewer than 200 lines we will not inlcude that character
for character in character_line_counts:
    if character[1] >= 200:
        most_common_characters.append(character[0])
simpsons_df = simpsons_df[simpsons_df['character_id'].isin(most_common_characters)]

temp_dict = {character: character_dict[character] for character in most_common_characters}
character_dict = temp_dict


# In[5]:


# Count how many lines are at location
locations = Counter(simpsons_df['location_id']).most_common()

most_common_locations = []

# If there are fewer than 100 lines at location we do not use that location
for location in locations:
    if location[1] >= 100:
        most_common_locations.append(location[0])
simpsons_df = simpsons_df[simpsons_df['location_id'].isin(most_common_locations)]

temp_dict = {location: location_dict[location] for location in most_common_locations}
location_dict = temp_dict


# In[6]:


# Here we count how many sentences each character has 
value_counts = simpsons_df['character_id'].value_counts()
# Plot a bar plot with the counts
count_plot = value_counts.plot(kind='bar', figsize=(15,8))
# Change the ID of thoose characters to names
count_plot.set_xticklabels([character_dict[i] for i in value_counts.keys()])
plt.show()


# In[7]:


simpsons_df['character_id'].value_counts(normalize=True)


# In[8]:


simpsons_df.head()


# In[9]:


# GOTT FYRIR SKÝRSLU: NOTUM TfidfVectorizer útaf því það er nákvæmara en CountVectorizer
# CountVectorizer telur bara frequncy á orðunum en TfidfVectorizer gerir mun á hversu frequent orð eru
# Nánara info hér: https://www.quora.com/What-is-the-difference-between-TfidfVectorizer-and-CountVectorizer-1

# Min_df is the minimum amount of toke frequency to be kept
# stop_words is to skip words like 'a, the' etc.
# sublinear_tf is used so we can do sublinear tf scaling (replace tf with 1+log(tf))
# Norm L2 is L2-normalization to minimize single outliers
# ngram_range is 1,3 so we select tokens of wordsizes: 1, 2 and 3

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, stop_words='english', norm='l2', ngram_range=(1,3))

features = tfidf.fit_transform(simpsons_df.normalized_text).toarray()
labels = simpsons_df.character_id


# In[ ]:


most_popular_df = pd.DataFrame()
# Get the 5 most popular tokens
token_count = 5
tokenNames = ['Token ' + str(i) for i in range(1, token_count+1)]

for character_id in character_dict:
    features_chi2 = chi2(features, labels == character_id)
    # Here the we sort the most common features of each character in order
    # features_chi2[0] stands for chi statistics of each character
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    # Split features in single or two word grams
    unigrams = [word for word in feature_names if len(word.split(' ')) == 1]
    biggrams = [words for words in feature_names if len(words.split(' ')) == 2]
    
    # If we don't have enough tokens for any charater we fill it with nan 
    if len(unigrams) < token_count:
        for i in range(len(unigrams), token_count+1):
            unigrams.append(np.nan)
    if len(biggrams) < token_count:
        for i in range(len(biggrams), token_count+1):
            biggrams.append(np.nan)
    
    # Append most common single word tokens to the dataframe
    character = "UNI: " + character_dict[character_id]
    most_popular_df = most_popular_df.append(
        pd.DataFrame([unigrams[-token_count:]], index=[character], columns=tokenNames))
    # Append the most common two word tokens to the dataframe
    character = "BIG: " + character_dict[character_id]                                          
    most_popular_df = most_popular_df.append(pd.DataFrame([biggrams[-token_count:]], index=[character], columns=tokenNames))
                                               
most_popular_df.head(10)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(simpsons_df['normalized_text'], simpsons_df['character_id'], random_state=42, test_size=0.2)
x_train_vec = tfidf.fit_transform(x_train)

mnb = MultinomialNB()
mnb.fit(x_train_vec, y_train)


# In[ ]:


x_test_vec = tfidf.transform(x_test)
test_prediction = mnb.predict(x_test_vec)
print("The accuracy is:", accuracy_score(y_test, test_prediction))

