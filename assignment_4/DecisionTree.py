
def information_gain(my_data, set1, set2):
    initial = entropy(my_data)
    return initial -len(set1)*entropy(set1)/len(my_data) - len(set2)*entropy(set2)/len(my_data)


# In[16]:

def split_dataset(dataset, column, value):
    if isinstance(value,int) or isinstance(value,float):
        df1 = dataset[dataset[column]>=value]
        df2 = dataset[dataset[column]< value]
    else:
        df1 = dataset[dataset[column]== value]
        df2 = dataset[dataset[column]!= value]
    return (df1,df2)


# In[66]:

def unique_counts(dataset, class_column="survived"):
    return dataset[class_column].value_counts().to_dict()


# In[67]:

from math import log
def entropy(dataset):
    log2 = lambda x: log(x)/log(2)
    results = unique_counts(dataset)
    ent = 0.0
    for k,v in results.items():
        p = float(v)/len(dataset)
        ent = ent - p*log2(p)
    return ent


# In[73]:

class tree_node:
    def __init__(self,col=-1,value=None,leftn=None, rightn=None, leaf=None):
        self.col=col
        self.value=value
        self.leftn=leftn
        self.rightn=rightn
        self.leaf=leaf


# In[174]:

def fit(features, target, score_f=entropy):
    label = target.name
    features[label] = target.values
    return build_tree(features, label, score_f)
def predict(features, tree):
    predictions = []
    for row in range(0,len(features)):
        predictions.append(classify(features.loc[row], tree))
    return predictions
def classify(features, tree):
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
        return classify(features, branch)
def build_tree(dataset,label="survived",score_f=entropy):
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
                # print("GAIN: {}, COL: {}, VAL: {}".format(infg,col,val))
                best_gain = infg
                best_col_val = (col,val)
                best_dfs = (df1,df2)
    if best_gain>0:
        leftn = build_tree(best_dfs[0])
        rightn = build_tree(best_dfs[1])
        return tree_node(best_col_val[0], best_col_val[1], leftn, rightn)
    else:
        return tree_node(leaf=unique_counts(dataset))


# In[110]:

# test1 = DecisionTree.fit(my_data[["source", "country", "yes_no", "pages"]], my_data["tier"])


# In[108]:

# printtree(test1)


# In[113]:

def printtree(tree,indent=''):
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


# In[175]:

# predict(my_data[["source", "country", "yes_no", "pages"]], test1)


# In[ ]:



