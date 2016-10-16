# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:37:37 2016

@author: fc20042008
"""
import pandas
import numpy

data = pandas.read_csv("/Users/fc20042008/Downloads/ML_Classification_2/amazon_baby.csv")
words = pandas.read_json("/Users/fc20042008/Downloads/ML_Classification_2/important_words.json")
pandas.Series(words[0]).tolist()

def remove_punctuation(text):
    import string
    replace = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(replace)

data = data.fillna({'review':''})  
data['rnew']  = data['review'].apply(remove_punctuation)
data['sentiment'] = data['rating'].apply(lambda x :1 if x > 3  else 0)
text = data['review']
text_final = []
for i in range(0,len(data['review'])):
    text_final.append(remove_punctuation(data['review'][i]))
    
for i in range(0,len(words)):
    data[words[0][i]] = data['rnew'].apply(lambda s: s.split().count(words[0][i]))
    
count = 0
for i in data['perfect']:
    if i > 0 :
        count = count + 1
        
def get_numpy_data(dataframe,features,label):
    dataframe['constant'] = 1
    features = ['constant'] + features    
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return (feature_matrix,label_array)

features = words[0].tolist()
feature_matrix , label_array = get_numpy_data(data,features,'sentiment')
    
def sigmoid(x):
    return (1/(1+numpy.exp(-x)))
def predict_probability(feature_matrix,coefficient):
    result = []
    d = numpy.dot(feature_matrix,coefficient)
    result = sigmoid(d)
    return (result)
    
def feature_derivative (error,feature_matrix):
    return (numpy.dot(numpy.transpose(feature_matrix,error)))
    
def compute_log_likelihood(feature_matrix,sentiment,coefficient):
    indicator = (sentiment == 1)
    scores = numpy.dot(feature_matrix,coefficient)
    logprob = numpy.sum((indicator-1)*scores - numpy.log(1.+numpy.exp(-scores)))
    return (logprob)
    
'''logistic regression with gradient descent'''
def logistic_regression(feature_matrix,sentiment,intitial_coefficients,step_size,max_iter):
    coeff = numpy.array(intitial_coefficients)
    for itr in range(max_iter):
        
        predictions = predict_probability(feature_matrix,coeff)
        indicator = (sentiment == 1)
        errors = predictions - indicator
        derivatives = feature_derivative(errors,feature_matrix)
        coeff = step_size * derivatives
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coeff)
            print ('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(numpy.ceil(numpy.log10(max_iter))), itr, lp))
    return coeff
    
coeff_test = logistic_regression(feature_matrix,data['sentiment'],numpy.zeros(194),1e-7,301)