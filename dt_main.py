from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call
from IPython.display import Image
import pandas as pd
import numpy as np
#data loading
dataset = pd.read_csv('./formatted_dataset/data_net_3.csv')
dataset.dofall = dataset['dofall'].replace('yes',1)
dataset.dofall = dataset['dofall'].replace('no',0)

X=np.array(pd.DataFrame(dataset, columns=['meanV','meanM','stdV','stdM']))
y=np.array(pd.DataFrame(dataset, columns=['dofall']))

#data split
X_train, X_test, y_train, y_test = train_test_split(X,y)

#train & evaluation
falltree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
falltree.fit(X_train, y_train)
y_pred = falltree.predict(X_test)

print('Accuracy: %.2f'%accuracy_score(y_test,y_pred))

#visualizaion

feature_names = dataset.columns.tolist()
feature_names = ['meanV','meanM','stdV','stdM']
target_name=np.array(['notfall','fall'])

export_graphviz(falltree, out_file='./graph/falltree_correct_4.dot',feature_names = feature_names,class_names = target_name)