# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:20:23 2016

@author: fc20042008
"""#{('name':np.str,'review':np.str,'rating':np.int)}
import numpy as np
#from io import StringIO
#file = open("/Users/fc20042008/Downloads/amazon_baby.csv",'rb')
#data = np.loadtxt(StringIO(file),delimiter=',',dtype=(np.str,np.str,np.str))
#data = np.genfromtxt('/Users/fc20042008/Downloads/amazon_baby.csv',delimiter=',')
#df=pd.read_csv('myfile.csv', sep=',',header=None)
#import csv
#data = None
#with open("/Users/fc20042008/Downloads/amazon_baby.csv",'r') as f:
#    rows = csv.reader(f)
#    data = np.array(rows)
#    #for row in rows:
#    #    print(row[0])
##print (data.shape)
#print(data.size)
#data.reshape(int(data.size  / 3), 3)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model



def remove_punctuation(text):
    import string
    replace = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(replace)
    
def sigmoid(x):
    import math
    return 1/(1+math.exp(-x))    
    
def correctness(predict,true,colname):    
    correct = 0 
    for i in range(0,len(predict)):
        if predict[i] == true.iloc[i,][colname]:
            correct = correct + 1
    return (correct*1.0/len(predict))  
    
    
    
df = pd.read_csv("/Users/fc20042008/Downloads/amazon_baby.csv")
text = df['review']
#print(df.select_dtypes(include=['float']))
#print (text[type(text)== float])

text.fillna('', inplace = True)    
text_final = []
#text = text.apply(remove_punctuation)
for i in range(0,100):
    try:
        pre_text = text[i] 
        #print (type(pre_text))
        content = remove_punctuation(pre_text)    
        text_final.append(content)
    except:
        print(text[i])
        print(i)
        print(type(text[i]))
        
df = df[df['rating']!=3]
df['sentiment'] = df['rating'].apply(lambda x :+1 if x> 3 else -1)
train_data = df.sample(frac = 0.8,random_state = 1)
test_data = df.drop(train_data.index)

vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review'])        
test_matrix = vectorizer.transform(test_data['review'])    
reg = linear_model.LogisticRegression()
model = reg.fit(train_matrix,train_data['sentiment'])
coeff = model.coef_

sample_test =test_data.iloc[10:13,]
sample_test_matrix = vectorizer.transform(sample_test['review'])
model.predict(sample_test_matrix)
result = model.predict(test_matrix)
score = model.decision_function(test_matrix)

prob = []
for i in score:
    prob.append(sigmoid(i))
    
    

print (correctness(result,test_data,'sentiment'))
    
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset = CountVectorizer(vocabulary = significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review']) 
reg2 = linear_model.LogisticRegression()
model2 = reg.fit(train_matrix_word_subset,train_data['sentiment'])
result2 = model2.predict(test_matrix_word_subset)
print (correctness(result2,test_data,'sentiment'))




