# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 10:39:02 2016

@author: fc20042008
"""

import pandas as pd
from sklearn import tree 
import numpy as np
from sklearn import preprocessing
df = pd.read_csv("/Users/fc20042008/Downloads/ML_Classification_3/lending-club-data.csv")
df = df.fillna({'desc':''})
df['safe_loans'] = df['bad_loans'].apply(lambda x: 1 if x==0 else -1)
df.shape   #(122607, 69)
safe = df.loc[df['safe_loans']==1,:]
safe.shape #(99457, 69)

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]
           
target = 'safe_loans'

loans = df[features+[target]]
train_ix = pd.read_json("/Users/fc20042008/Downloads/ML_Classification_3/module-5-assignment-1-train-idx.json")
validate_ix = pd.read_json("/Users/fc20042008/Downloads/ML_Classification_3/module-5-assignment-1-validation-idx.json")
train_index = []
for i in range(len(train_ix)):
    train_index.append(train_ix.iloc[i][0])
validate_index = []
for i in range(len(validate_ix)):
    validate_index.append(validate_ix.iloc[i][0])   
    
categorical_variables = []  
for feat_name in features:
    if loans[feat_name].dtype == 'object':
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    one_hot_feature = pd.get_dummies(loans[feature])
    loans = loans.join(one_hot_feature)
    loans.drop(feature,inplace = True,axis = 1)

    
train_data = loans.iloc[train_index,:]
validation_data = loans.iloc[validate_index]

safe_loans = train_data.loc[train_data.loc[:,'safe_loans']==1,:]


    
y = train_data.loc[:,'safe_loans']
train_data.drop(['safe_loans'],inplace = True,axis = 1)    
features = train_data.columns
dt = tree.DecisionTreeClassifier(max_depth = 3)
dt = dt.fit(train_data,y)
t_y = dt.predict(train_data)
tree.export_graphviz(dt)

validation_y = validation_data.loc[:,'safe_loans']
validation_data.drop(['safe_loans'],inplace = True,axis = 1)   
predict_y = dt.predict(validation_data) 
percent_t = np.sum(y == t_y)/len(t_y)
percent_v = np.sum(validation_y == predict_y)/len(predict_y)
print ("train accuracy: %.2f" % percent_t)  #0.62
print("predict accuracy: %.2f" % percent_v)  #0.62
    