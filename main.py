'''
Step 1 Data Preprocessing
'''
import numpy as np
import pandas as pd
from collections import defaultdict
import csv

original_data = pd.read_csv('data.csv', encoding='cp1252')
data = original_data

data = data.fillna(method='ffill')

'''
Here we have created a dictionary which maps a disease to all of its symptoms as a list
'''

def cleaner_function(item):
    item_list = []
    match = item.replace('^', '_').split('_')
    count = 1
    for name in match:
        if count%2==0:
            item_list.append(name)
        count = count +1
    return item_list

disease_name = ""
disease_wt = 0
disease_list = []
dict_wt = {}
disease_dict = defaultdict(list)
for i in range(len(data)):
    row0 = data.iloc[i, 0]
    row2 = data.iloc[i, 2]
    row1 = data.iloc[i, 1]
    if row0!="\xc2\xa0" and row0!="":
        disease_name = row0
        disease_wt = row1
        disease_list = cleaner_function(disease_name)
        
    if row2!="\xc2\xa0" and row2!="":
        symptom_list = cleaner_function(row2)
        for name in disease_list:
            for symptom in symptom_list:
                if symptom!="" and symptom!=" ":
                    disease_dict[name].append(symptom)
            dict_wt[name] = disease_wt

'''
Here we have created a Saaf_Data from disease dict 
'''
with open("Saaf_data.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    for key, value in disease_dict.items():
        for symptoms in value:
            writer.writerow([key, symptoms, dict_wt[key]])
            
saaf_data = pd.read_csv('Saaf_data.csv', header=None)
columns = ['Disease', 'Symptom', 'Occurrence']
saaf_data.columns = columns

'''
here we have one hot encoded categorical variables
'''

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
int_encoded = label_encoder.fit_transform(saaf_data['Symptom'])

one_hotencoder = OneHotEncoder(sparse=False)
int_encoded = int_encoded.reshape(len(int_encoded), 1)
one_hotencoded = one_hotencoder.fit_transform(int_encoded)

'''
we are going to map symptomps to their codes basically their indexes in one_hotencoded using a dict
'''
symptom_mapping = {}

for i in range(len(saaf_data)):
    key = int_encoded[i][0]
    val = saaf_data.iloc[i, 1]
    symptom_mapping[key] = val

'''
here we are changing the names of one_hotencoded and making a dataframe out of it
'''
#we have made array of symptoms in the same order as they are in one_hotencoded
cols = []
for i in range(len(one_hotencoded[0])):
    cols.append(symptom_mapping[i])  
    
#we have made a new dataframe

f_data = pd.DataFrame(one_hotencoded)
f_data.columns = cols

'''
next is the final step to create our training_data.csv
'''

saaf_data_disease = saaf_data['Disease']
f_data = pd.concat([saaf_data_disease, f_data], axis = 1)
f_data.drop_duplicates(keep='first', inplace=True)

f_data = f_data.groupby('Disease').sum()
f_data = f_data.reset_index()

#finally we exported training data
f_data.to_csv(r'training_data.csv', header=True)

'''
DECISION TREE CLASSIFIER
'''
X1 = X = f_data.iloc[:, 1:]
X = f_data.iloc[:, 1:].values
y = f_data.iloc[:, 0].values

X1.to_csv(r'test_set.csv', header=False)
from sklearn.tree import DecisionTreeClassifier
'''
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X, y)

y_pred = classifier.predict(X)
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y, y_pred)
score = accuracy_score(y, y_pred)
'''

'''
PART 2 PIPELINE
'''

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('model_name', DecisionTreeClassifier(criterion='entropy', random_state=0))])
pipeline.fit(X, y)
y_pred = pipeline.predict(X)
score = pipeline.score(X, y)

'''
PART 3 PIPELINE DUMP
'''

from joblib import dump
dump(pipeline, filename="disease_predictor.joblib")
