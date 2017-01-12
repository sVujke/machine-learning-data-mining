
# coding: utf-8

# # MLDM Group Assignment
# ### Members: Brigitte Aznar, Daniel Kostic and Stefan Vujovic

# Data explanation
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# 

# In[1]:

import pandas as pd
import numpy as np
df = pd.read_csv('data.csv', index_col=0)
# df.survived = df.survived.apply(lambda x: "yes" if x else "no")
# df.to_csv('data1.csv')


# In[2]:

# df['sex1'] = df.sex.apply(lambda x: 1 if x=="male" else 0)
# df['survived'] = df.survived.apply(lambda x: "survived" if x==1 else "")
def makeClass(x):
    if x==1: return "firstClass"
    elif x==2: return "secondClass"
    else: return "thirdClass"
df['pclass'] = df.pclass.apply(lambda x: makeClass(x))
df.fare = df.fare.apply(lambda x: 33 if pd.isnull(x) else x)
# df.fare.dropna(inplace=True, how='any')
# df.embarked.dropna(inplace=True)
df.head()


# In[3]:

df.embarked.isnull().value_counts()


# In[4]:

df.age.describe()
mean = 29.881135
std_dev = 14.413500
age_fillers = np.random.normal(mean, std_dev, 280)
age_fillers = [i for i in age_fillers if i>0][:263]
df['age1'] = df.age.apply(lambda x: age_fillers.pop() if np.isnan(x) else x)


# In[5]:

cat_columns = ['pclass', 'sex'] # embarked
for cat_col in cat_columns:
    df = df.join(pd.get_dummies(df[cat_col]))


# In[6]:

from sklearn.model_selection import train_test_split


# In[7]:

from sklearn import tree
features = ['age1', 'firstClass', 'secondClass', 'thirdClass', 'sibsp', 'parch', 'fare', 'female'] #, 'C', 'Q', 'S'
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(df[features], df['survived'])


# In[8]:

tree.export_graphviz(clf, out_file='tree.dot')


# In[9]:

for feat in features:
    print("{}, from {}".format(df[feat].isnull().sum(), feat))


# In[10]:

df.head()


# In[11]:

def gini(attribute):
    att_values = df[attribute].unique()
    for class_val in [0,1]:
        for val in att_values:
            size = df[df[attribute]==val].shape[0]
            gini = df[df[attribute]==val].survived.count() / float(size)
            print ("Class:{}, Attribute:{}, GINI:{}".format(class_val, val, gini))


# In[12]:

gini('sex')


# In[13]:

[1,2,3,3,3,3,3].count(3)


# In[14]:

df.head()


# In[15]:

from DecisionTree import DecisionTree as Dt


# In[16]:

dt = Dt()
dt.fit(df[['sex', 'age1', 'pclass', 'fare']], df['survived'])


# In[17]:

predictions = dt.predict(df[['sex', 'age1', 'pclass', 'fare']])


# In[18]:

df['predictions'] = pd.Series(predictions)


# In[19]:

df['diff'] = df['survived'] - df['predictions']


# In[20]:

df['diff'].sum()


# In[21]:

df[['survived', 'predictions', 'diff']].head()


# In[24]:

from DecisionTree import bagging
from sklearn.model_selection import train_test_split
from time import time
now = time()
x_train, x_test, y_train, y_test = train_test_split(df.drop('survived', 1), df['survived'], test_size=0.2)
x_train['survived']  = y_train
x_test['survived'] = y_test

preds = bagging(x_train, x_test, ['sex', 'age1', 'pclass', 'fare'], 'survived', 9)
print(now - time())


# In[ ]:

preds


# In[ ]:



