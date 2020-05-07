#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import SVG
from IPython.display import display
from sklearn import tree
from ipywidgets import interactive
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB


# In[4]:


matches = pd.read_csv("matches.csv")
matches.dtypes


# In[5]:


matches


# In[6]:


l = list(matches['season'])
max=2019
min=2019
for i in l:
  if(i<min):
    min=i
  if(i>max):
    max=i

print("max year is",max)
print("min year is",min)


# In[7]:


# No. of mathces played in each season

cnt = [0,0,0,0,0,0,0,0,0,0,0,0]
year=2008
for i in l:
  cnt[i-2008] = cnt[i-2008]+1

for i in cnt:
  print(year," ",i)
  year+=1


# In[8]:


# All the TEAMS

all_teams = set({})
j=0
for i in l:
  all_teams.add(matches['team1'][j])
  all_teams.add(matches['team2'][j])
  j+=1

print(all_teams)
print(len(all_teams))


# In[9]:


matches['winner'].fillna("draw", inplace = True)


# In[12]:


# Labelling the Teams and Winner

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
all_teams.add('draw')
le.fit(list(all_teams))
le.classes_
matches['winner']=le.transform(matches['winner'])
matches


# In[11]:


# Labelling the Toss_decision and Toss_winner

from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()

le2.fit(['field','bat'])
le2.classes_
matches['toss_winner']=le.transform(matches['toss_winner'])
matches['toss_decision']=le2.transform(matches['toss_decision'])
matches


# In[13]:


venues = set({})
j=0
for i in l:
  venues.add(matches['venue'][j])
  j+=1

print(venues)
print(len(venues))


# In[14]:


# Labelling the venues

from sklearn import preprocessing
le3 = preprocessing.LabelEncoder()
v = list(venues)
le3.fit(v)
le3.classes_
matches['venue']=le3.transform(matches['venue'])
matches


# In[15]:


total=0
match_and_toss_won=0
for i in range(756):
  if(i not in matches.index):
    continue
  total+=1
  if(matches['winner'][i]==matches['toss_winner'][i]):
    match_and_toss_won+=1
  

print(" Percentage of winning both match and toss : ",match_and_toss_won*100/total)


# In[16]:


played = np.zeros(15)
won = np.zeros(15)

for i in range(756):
  if(i not in matches.index):
    continue
  played[matches['team1'][i]]+=1
  played[matches['team2'][i]]+=1
  if(matches['winner'][i]==matches['team1'][i]):
    won[matches['team1'][i]]+=1
  elif(matches['winner'][i]==matches['team2'][i]):
    won[matches['team2'][i]]+=1

for i in np.arange(15):
    print(i+1,'   ',(le.inverse_transform([i])[0]),"\t\t")


# In[17]:


# Getting Winning Percentage of all Teams

win_age = np.zeros(15)
for i in range(15):
  win_age[i]=(won[i]/played[i])*100

print("Team\t\tWon%Age")
print()
for i in np.arange(15):
    print((le.inverse_transform([i])[0]),"\t\t")
    print("\t\t",win_age[i])


# In[18]:


# Getting Toss and Match win Percentage

toss_and_match_win = np.zeros(15)

for i in range(756):
  if(i not in matches.index):
    continue
  elif(matches['winner'][i]==matches['team1'][i] and matches['toss_winner'][i]==matches['team1'][i]):
    toss_and_match_win[matches['team1'][i]]+=1
  elif(matches['winner'][i]==matches['team2'][i] and matches['toss_winner'][i]==matches['team2'][i]):
    toss_and_match_win[matches['team2'][i]]+=1



toss_and_match_win_age = np.zeros(15)

for i in range(15):
  toss_and_match_win_age[i] = (toss_and_match_win[i]/played[i])*100

for i in np.arange(15):
    print((le.inverse_transform([i])[0]),"\t\t")
    print("\t\t",toss_and_match_win_age[i])


# In[19]:


matches['win_by_wickets']*=5
matches['team1_wins']=(matches['winner']==matches['team1']).astype(int)
matches['team2_wins']=(matches['winner']==matches['team2']).astype(int)


# In[20]:


matches['team1_wins_toss']=(matches['toss_winner']==matches['team1']).astype(int)
matches['team2_wins_toss']=(matches['toss_winner']==matches['team2']).astype(int)


# In[21]:


matches['season']-=2007
matches['toss_bat']=(matches['toss_decision']==0).astype(int)
matches['toss_field']=(matches['toss_decision']==1).astype(int)


# In[22]:


del matches['toss_decision']


# In[23]:


del matches['toss_winner']


# In[24]:


matches.corr()


# In[25]:



X=matches.loc[:,['season','win_by_runs','win_by_wickets','team1','team2','team1_wins_toss','team2_wins_toss','toss_bat','toss_field','venue']]
y=matches['team1_wins']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[26]:



knn5 = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn5.fit(X_train, y_train)

#Predict the response for test dataset
knn5_predict = knn5.predict(X_test)


# In[27]:



from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, knn5_predict))


# In[28]:


knn5_auc=roc_auc_score(y_test,knn5_predict)


# In[29]:


print(knn5_auc)


# In[42]:



a = knn5.predict([['13', '0', '9', '7', '8', '0', '1', '0', '1', '40']])
print(a)


# In[ ]:




