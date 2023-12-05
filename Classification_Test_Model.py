#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Preprocessing for model consumption component

csvFilePath = "classification_training_set.csv"
import pandas as pd


# Read the CSV file into a pandas dataframe
column_names = ['Article Title', 'Classification', 'Date']
dataframe = pd.read_csv(csvFilePath, header=None, names = column_names)


# Display the first few rows of the dataframe to verify the data has been loaded properly
print(dataframe.head())


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Assuming 'X' contains article titles and 'y' contains corresponding labels
X = dataframe['Article Title']
y = dataframe['Classification']
# Splitting data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Training a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predicting on the validation set
predictions = clf.predict(X_val_tfidf)

# Evaluating the model
print(classification_report(y_val, predictions))



# In[3]:


# Read the new data
testCSV = "Headlines2022_Sentiment_Analysis.csv" # replace with your file path
new_dataframe = pd.read_csv(testCSV)

# Vectorizing new data and predicting
X_new_tfidf = tfidf_vectorizer.transform(new_dataframe['Title'])
new_predictions = clf.predict(X_new_tfidf)

# Writing predictions to CSV
new_dataframe['Predicted Classification'] = new_predictions
new_dataframe.to_csv("Predictions.csv", index=False)


# In[ ]:




