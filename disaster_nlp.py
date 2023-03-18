#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
data_frame = pd.read_csv('tweets_mod_4.csv')


# In[25]:


import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re


# In[26]:


nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')


def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

#### Read and remove the unwanted cols
#pd.set_option('display.max_colwidth', None)
cols_to_drop = ['id']
train_data = data_frame.drop(cols_to_drop, axis=1)
train_data.head()
print(train_data)

### Clean the text using Re
data_clean = clean_text(train_data, 'text', 'text_1_clean')
data_clean.head()


### Stop words
stop = stopwords.words('english')
data_clean['text_1_clean'] = data_clean['text_1_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_clean.head()


## Tokenize
data_clean['text_tokens'] = data_clean['text_1_clean'].apply(lambda x: word_tokenize(x))
data_clean.head()

def word_stemmer(text):
    stem_text = [PorterStemmer().stem(i) for i in text]
    return stem_text
data_clean['text_tokens_stem'] = data_clean['text_tokens'].apply(lambda x: word_stemmer(x))
data_clean.head()


def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text
data_clean['text_tokens_lemma'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))
data_clean.head()


def word_pos_tagger(text):
    pos_tagged_text = nltk.pos_tag(text)
    return pos_tagged_text
nltk.download('averaged_perceptron_tagger')
data_clean['text_tokens_pos_tagged'] = data_clean['text_tokens'].apply(lambda x: word_pos_tagger(x))
data_clean.head()

data_clean.keyword.value_counts().plot(kind='bar');


# In[28]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
X = data_clean.text_1_clean
y = data_clean.keyword
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
nb = Pipeline([('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB()),
])
nb.fit(X_train, y_train)
from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
my_tags=['flood','fire','earthquake','cyclone']#,'snow','landslide','drought','volcano','lightning','tsunami'];
print(classification_report(y_test, y_pred,target_names=my_tags))
final_list=[]


from location_extractor import extract_locations
#text = "I arrived in New York on January 4, 1937"
#locations = extract_locations(text)
#print(locations)
from geotext import GeoText

import json
for ind in data_clean['text']:
  temp_list=[]
  temp_list.append(ind)
  #print(ind)
  input=[ind]
  temp= nb.predict(input);
  #print (temp,"...temp")
  if temp in my_tags:
    locations = extract_locations(ind)
    if (len(locations) >0):
      temp_list.append(locations)
    else:
      places = GeoText(ind)
      if(len(places.cities)>0):
        temp_list.append(places.cities)
    temp_list.append(temp[0])
    final_list.append(temp_list)


training_insert_query = """ INSERT INTO training_table (name, coordinate, tweet,type) VALUES (%s,ST_GeomFromText(%s),%s,%s)"""
#testing_insert_query = """ INSERT INTO testing_table (name, coordinate, tweet,type) VALUES (%s,%s,%s,%s)"""
import requests 
import psycopg2
try:
    connection = psycopg2.connect(user="postgres",password="krishna1736",host="localhost",port="5433",database="postgres")
    cursor = connection.cursor()
    # api-endpoint
    for i in final_list:
      #print(len(i)," len of final list")
      #print(i[1],"...2nd")
      if(len(i)>1 and len(i[1])==1):
        listToStr = ' '.join([str(elem) for i,elem in enumerate(i[1])])
        #print(listToStr)
        URL = "https://dev.virtualearth.net/REST/v1/Locations/"+listToStr+"?o=json&key=Av6xTqKcig2Bj33O39PiWJdDcktUYfLHZC9Opq77_bzuhmToeKTgERJOh97kIyF8&includeNeighborhood=1&maxResults=1"
        # location given here
        # sending get request and saving the response as response object
        r = requests.get(url = URL)
        # extracting data in json format
        data = r.json()
        #y = json.loads(data)
        #print(data)
        #print(listToStr)
        if len(data["resourceSets"][0]["resources"]) >0:
          confi = data["resourceSets"][0]["resources"][0]["confidence"]
          if (confi =="High"):
            point= data["resourceSets"][0]["resources"][0]["point"]["coordinates"]
            point = "Point("+str(point).replace('[','').replace(']','').replace(',','')+")"
            #print (type(point))
            #print(type(i[0]))
            cursor.execute(training_insert_query,(listToStr,str(point),i[0],i[2]))
            connection.commit()
    print('Training Data has been inserted')
except (Exception, psycopg2.Error) as error:
    print("Failed to insert record into training  table", error)
finally:
    # closing database connection.
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
 
testing_insert_query = """ INSERT INTO testing_table (name, coordinate, tweet,type) VALUES (%s,ST_GeomFromText(%s),%s,%s)"""
try:
    connection = psycopg2.connect(user="postgres",password="krishna1736",host="localhost",port="5433",database="postgres")
    cursor = connection.cursor()  
    final_list1=[]  
    for ind in X_test:
      temp_list=[]
      temp_list.append(ind)
      #print(ind)
      input=[ind]
      temp= nb.predict(input);
      #print (temp,"...temp")
      if temp in my_tags:
        locations = extract_locations(ind)
        if (len(locations) >0):
          temp_list.append(locations)
        else:
          places = GeoText(ind)
          if(len(places.cities)>0):
            temp_list.append(places.cities)
        temp_list.append(temp[0])
        final_list1.append(temp_list)
    # api-endpoint
    for i in final_list1:
      #print(len(i)," len of final list")
      #print(i[1],"...2nd")
      if(len(i)>1 and len(i[1])==1):
        listToStr = ' '.join([str(elem) for i,elem in enumerate(i[1])])
        #print(listToStr)
        URL = "https://dev.virtualearth.net/REST/v1/Locations/"+listToStr+"?o=json&key=Av6xTqKcig2Bj33O39PiWJdDcktUYfLHZC9Opq77_bzuhmToeKTgERJOh97kIyF8&includeNeighborhood=1&maxResults=1"
        # location given here
        # sending get request and saving the response as response object
        r = requests.get(url = URL)
        # extracting data in json format
        data = r.json()
        #y = json.loads(data)
        #print(data)
        #print(listToStr)
        if len(data["resourceSets"][0]["resources"]) >0:
          confi = data["resourceSets"][0]["resources"][0]["confidence"]
          if (confi =="High"):
            point= data["resourceSets"][0]["resources"][0]["point"]["coordinates"]
            point = "Point("+str(point).replace('[','').replace(']','').replace(',','')+")"
           
            #print(point)           #print (type(point))
            #print(type(i[0]))
            #print(testing_insert_query,(listToStr,point,i[0]))
            cursor.execute(testing_insert_query,(listToStr,point,i[0],i[2]))
            connection.commit()
    print('Testing Data has been inserted') 
except (Exception, psycopg2.Error) as error:
    print("Failed to insert record into test  table", error)
   
finally:
    # closing database connection.
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed") 


# In[ ]:




