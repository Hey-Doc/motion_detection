from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#data loading
dataset = pd.read_csv('./formatted_dataset/data_net_2.csv')
dataset.dofall = dataset['dofall'].replace('yes',1)
dataset.dofall = dataset['dofall'].replace('no',0)

X=np.array(pd.DataFrame(dataset, columns=['meanV','meanH','meanM','stdV','stdH','stdM']))
y=np.array(pd.DataFrame(dataset, columns=['dofall']))

y=y.reshape(-1,) 

#data split
X_train, X_test, y_train, y_test = train_test_split(X,y)

#train & evaluation
num=4
fallfst = RandomForestClassifier(criterion='entropy', n_estimators = num,max_depth=3, random_state=1)
fallfst.fit(X_train, y_train)
y_pred = fallfst.predict(X_test)

print('Accuracy: %.2f'%accuracy_score(y_test,y_pred))
"""
#visualizaion
feature_names = dataset.columns.tolist()
feature_names = feature_names[1:7]
target_name=np.array(['notfall','fall'])

for i in range(num):
    estimator = fallfst.estimators_[i]
    export_graphviz(estimator, out_file='./graph/fallfsttree_%d.dot' %i, feature_names = feature_names,class_names = target_name)
    
"""