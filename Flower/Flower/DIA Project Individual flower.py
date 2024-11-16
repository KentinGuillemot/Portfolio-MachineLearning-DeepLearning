#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns


flower = pd.read_csv('IRIS_ Flower_Dataset.csv')
flower.head(10)


# In[5]:


flower.shape


# In[6]:


flower.describe()


# In[7]:


#Separate the features and the targets 

X = flower.drop("species", axis = 1)
y = flower["species"]


# In[8]:


X.head()


# In[9]:


y.head()


# In[10]:


#Separate x and y dataframes into test and train dataframes 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)


# In[11]:


# Function to train and evaluate the accuracy, classification report and the matrice of confusion

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy : {accuracy * 100:.2f}%')
    
    print('\nClassification_report :')
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.title('Matrice de Confusion')
    plt.show()




# In[12]:


# The model random forest and his training with the importants inforrmations

rf_model = RandomForestClassifier(n_estimators=50, random_state=32)
train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test)


# In[13]:


# Second try of a random forest model but with other hyperparaleters

rf_modelV2 = RandomForestClassifier(n_estimators=50, random_state=42)
train_and_evaluate_model(rf_modelV2, X_train, y_train, X_test, y_test)


# In[14]:


# Obtain the importances of each feature in the training and the prediction
feature_importances = rf_modelV2.feature_importances_

# Obtain the index of the features in ascending order
sorted_indices = np.argsort(feature_importances)

# Creation of the graph
plt.figure(figsize=(8, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), X.columns[sorted_indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Importance of features in Model Predictions (Ascending order)')
plt.show()


# In[15]:


#Function to find what kind a flower we have we specific features

def flower():
    tab_test = []

    sepal_length = float(input("sepal_length : "))
    tab_test.append(sepal_length)
    
    sepal_width = float(input("sepal_width : "))
    tab_test.append(sepal_width)
    
    petal_length = float(input("petal_length : "))
    tab_test.append(petal_length)
    
    petal_width = float(input("petal_width : "))
    tab_test.append(petal_width)

    tab_test2 = []
    tab_test2.append(tab_test)

    data_test = pd.DataFrame(tab_test2) #crée une dataframe à partir d'un tableau

    pred = rf_modelV2.predict(data_test)
    
    
    print("\n{}".format(pred))


# In[16]:


flower()


# In[ ]:




