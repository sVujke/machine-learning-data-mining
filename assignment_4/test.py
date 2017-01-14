
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


# In[2]:

def makeClass(x):
    if x==1: return "firstClass"
    elif x==2: return "secondClass"
    else: return "thirdClass"
df['pclass'] = df.pclass.apply(lambda x: makeClass(x))

# df.fare.dropna(inplace=True, how='any')
df = df[pd.notnull(df['embarked'])]
df.head()


# ## What is the age/class of persons whose fare is 0? Are they babies/kids?

# In[3]:

df[df.fare==0].age.value_counts(dropna=False)


# In[4]:

df[df.fare==0].age.value_counts(dropna=False)


# In[5]:

df[df.fare==0].age.describe()


# As they are not kids, lets replace their fare with the mean.

# In[6]:

df.fare = df.fare.apply(lambda x: 33 if pd.isnull(x) else x)


# In[7]:

df.embarked.value_counts(dropna=False)


# In[8]:

df.age.describe()
mean = 29.881135
std_dev = 14.413500
age_fillers = np.random.normal(mean, std_dev, 280)
age_fillers = [i for i in age_fillers if i>0][:263]
df['age1'] = df.age.apply(lambda x: age_fillers.pop() if np.isnan(x) else x)
# df['age1'] = df.age.apply(lambda x: mean if np.isnan(x) else x)


# df = df[pd.notnull(df['age'])]
# df['age1'] = df.age


# In[9]:

cat_columns = ['pclass', 'sex', 'embarked'] # embarked
for cat_col in cat_columns:
    df = df.join(pd.get_dummies(df[cat_col]))


# In[10]:

from sklearn.model_selection import train_test_split


# In[11]:

# from sklearn import tree
# features = ['age1', 'firstClass', 'secondClass', 'thirdClass', 'sibsp', 'parch', 'fare', 'female', 'C', 'Q', 'S'] #, 'C', 'Q', 'S'
# clf = tree.DecisionTreeClassifier(criterion='gini')
# clf = clf.fit(df[features], df['survived'])


# In[12]:

# tree.export_graphviz(clf, out_file='tree.dot')


# In[13]:

# for feat in features:
#     print("{}, from {}".format(df[feat].isnull().sum(), feat))


# In[14]:

df.head()


# In[15]:

from DecisionTree import DecisionTree as Dt
from DecisionTree import bagging, generate_trees
from sklearn.model_selection import train_test_split
from time import time
x_train, x_test, y_train, y_test = train_test_split(df.drop('survived', 1), df['survived'], test_size=0.2)
x_train['survived']  = y_train
x_test['survived'] = y_test

features = ['sex', 'age1', 'pclass', 'fare', 'embarked', 'sibsp', 'parch']


# In[ ]:

now = time()

trees = generate_trees(x_train, features, 'survived')
print(now - time())


# In[ ]:

preds = bagging(x_train, x_test, features, 'survived', 9, trees)


# In[ ]:

print(len(trees))
print(len(x_test))
print(len(y_test))
print(len(preds))


# In[19]:

from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, preds))


# In[ ]:

df.survived.value_counts(normalize=True)


# In[22]:

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf_feat = ['age1', 'firstClass', 'secondClass', 'thirdClass', 'sibsp', 'parch', 'fare', 'female', 'C', 'Q', 'S']
clf = clf.fit(x_train[clf_feat], y_train)
clf_preds = clf.predict(x_test[clf_feat])
print(accuracy_score(y_test, clf_preds))


# In[ ]:

x_train.head()


# In[17]:

dt = Dt()
dt.fit(x_train[features], y_train)
predictions = dt.predict(x_test[features])
print(accuracy_score(y_test, predictions))


# age with random
# we - 76
# sk - 79
# bg - 81,6

# In[23]:

from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier(criterion='entropy')
scores = cross_val_score(clf, df[clf_feat], df['survived'], cv=10)


# In[ ]:

scores.mean()


# In[ ]:

scores


# In[ ]:

from sklearn.model_selection import KFold
def cross_val(classifier, features, label, num_iter):
    pass


# In[ ]:

kf = KFold(3)
print(kf.split(df))


# In[24]:

scores1 = cross_val_score(Dt(), df[features], df['survived'], cv=10)


# In[ ]:



