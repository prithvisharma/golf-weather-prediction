from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image

import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


golf_df = pd.DataFrame()

golf_df['Outlook'] = ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 
                     'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast',
                     'overcast', 'rainy']

golf_df['Temperature'] = ['hot',  'hot',  'hot',  'mild',  'cool',  
                           'cool',  'cool', 'mild',  'cool',  
                           'mild',  'mild',  'mild',  'hot',  'mild']

golf_df['Humidity'] = ['high', 'high', 'high', 'high', 'normal', 
                       'normal', 'normal', 'high', 'normal', 
                       
'normal', 'normal', 'high', 'normal', 'high']


golf_df['Windy'] = ['false', 'true', 'false', 'false', 'false', 
                   'true', 'true', 'false', 'false', 'false',
                   'true', 'true', 'false', 'true']

golf_df['Play'] = ['no', 'no', 'yes', 'yes', 'yes', 'no', 
                   'yes', 'no', 'yes', 'yes', 'yes', 
                  'yes', 'yes', 'no']

one_hot_data = pd.get_dummies(golf_df[['Outlook','Temperature','Humidity','Windy']])
#print(golf_df)
#print(one_hot_data)

clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(one_hot_data, golf_df['Play'])

dot_data = tree.export_graphviz(clf_train, out_file=None, 
            feature_names=list(one_hot_data.columns.values), 
        class_names=['Not_Play', 'Play'], rounded=True, filled=True) 
#print(dot_data)

graph = pydotplus.graph_from_dot_data(dot_data)

display(Image(graph.create_png()))

prediction = clf_train.predict([[0,0,1,0,1,0,0,1,1,0]])

print(prediction)