
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import accuracy_score
def split_dataset(dataset, column, value):
    if isinstance(value,int) or isinstance(value,float):
        df1 = dataset[dataset[column]>=value]
        df2 = dataset[dataset[column]< value]
    else:
        df1 = dataset[dataset[column]== value]
        df2 = dataset[dataset[column]!= value]
    return (df1,df2)


# In[7]:

def unique_counts(dataset, class_column="survived"):
    return dataset[class_column].value_counts().to_dict()


# In[8]:
def gini_impurity(dataset):
    total=len(dataset)
    counts=unique_counts(dataset)
    imp=0
    for k1 in counts:
        p1=float(counts[k1])/total
        for k2 in counts:
            if k1==k2: continue
            p2=float(counts[k2])/total
            imp+=p1*p2
    return imp


from math import log
def entropy(dataset):
    log2 = lambda x: log(x)/log(2)
    results = unique_counts(dataset)
    ent = 0.0
    for k,v in results.items():
        p = float(v)/len(dataset)
        ent = ent - p*log2(p)
    return ent


# In[9]:

class tree_node:
    def __init__(self,col=-1,value=None,leftn=None, rightn=None, leaf=None):
        self.col=col
        self.value=value
        self.leftn=leftn
        self.rightn=rightn
        self.leaf=leaf


# In[96]:

class DecisionTree:
    def __init__(self, tree=None):
        self.tree = tree
    def fit(self, features, target, score_f=entropy):
        label = target.name
        features[label] = target.values
        self.tree = self.build_tree(features, label, score_f)
    def predict(self, features):
        predictions = []
        for row in range(0,len(features)):
            predictions.append(self.classify(features.iloc[row], self.tree))
        return predictions
    def classify(self, features, tree):
        if tree.leaf != None:
            return list(tree.leaf.keys())[0]
        else:
            v = features[tree.col]
            branch = None
            if isinstance(v,int) or isinstance(v,float):
                if v>= tree.value:
                    branch = tree.leftn
                else:
                    branch = tree.rightn
            else:
                if v==tree.value:
                    branch = tree.leftn
                else:
                    branch = tree.rightn 
            return self.classify(features, branch)
    def build_tree(self, dataset,label="survived",score_f=entropy):
        if len(dataset) == 0:
            return tree_node()
        current_score = score_f(dataset)

        best_gain = 0.0
        best_col_val = None
        best_dfs = None

        columns = list(dataset.columns)
        columns.remove(label)
        for col in columns:
            unique_vals = list(dataset[col].unique())
            for val in unique_vals:
                (df1,df2) = split_dataset(dataset, col, val)
                p = float(len(df1)/len(dataset))
                infg = current_score - p*score_f(df1) - (1-p)*score_f(df2)
                if infg > best_gain and len(df1)>0 and len(df2)>0:
#                     print("GAIN: {}, COL: {}, VAL: {}".format(infg,col,val))
                    best_gain = infg
                    best_col_val = (col,val)
                    best_dfs = (df1,df2)
        if best_gain>0:
            leftn = self.build_tree(best_dfs[0])
            rightn = self.build_tree(best_dfs[1])
            return tree_node(best_col_val[0], best_col_val[1], leftn, rightn)
        else:
            return tree_node(leaf=unique_counts(dataset))
    def score(x, y):
        return accuracy_score(self.predict(x), y)
        
def printtree(tree, indent=''):

    # Is this a leaf node?
    if tree.leaf!=None:
        print(str(tree.leaf))
    else:
        print(str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->', end=" ")
        printtree(tree.leftn,indent+'  ')
        print(indent+'F->', end=" ")
        printtree(tree.rightn,indent+'  ')



def generate_trees(train, features, label, m=9):
    dt_list = []
    for i in range(0,m):
        sample = train.sample(len(train), replace=True)
        dt = DecisionTree()
        dt.fit(sample[features], sample[label])
        dt_list.append(dt)
    return dt_list




from collections import Counter
def bagging(train, test, features, label, m, dt_list):
    predictions = []
    df = pd.DataFrame()
    # Iterate over decision trees
    for i in range(0, len(dt_list)):
        dt = dt_list[i]
        df[i] = pd.Series(dt.predict(test))
    for row in df.iterrows():
        c = Counter([pred for pred in row[1]])
        predictions.append(c.most_common()[0][0])
        # print("Iter {}, Pred: {}", i, predictions)
    return predictions


import random
def generate_simple_trees(train, features, label, m=9, num_feats=3):
    dt_list = []
    for i in range(0,m):
        sample = train.sample(len(train), replace=True)
        random.shuffle(features)
        features[0:num_feats]
        dt = DecisionTree()
        dt.fit(sample[features], sample[label])
        dt_list.append(dt)
    return dt_list
