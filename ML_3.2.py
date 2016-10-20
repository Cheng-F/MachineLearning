# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:20:39 2016

@author: fc20042008
"""

import pandas as pd
import numpy as np
import math

df = pd.read_csv('lending-club-data.csv',low_memory = False)
df['safe_loans'] = df['bad_loans'].apply(lambda x : 1 if  x == 0 else -1)
df.drop(['bad_loans'],inplace = True,axis = 1)

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'


train_ix = pd.read_json('module-5-assignment-2-train-idx.json')
test_ix = pd.read_json('module-5-assignment-2-test-idx.json')

def transform_df_to_list(df):
    df_list = []    
    for i in range(df.shape[0]):
        df_list.append(df.iloc[i][0])
    return df_list

train_idx = transform_df_to_list(train_ix)
test_idx = transform_df_to_list(test_ix)
loans = df[features]
loans_target = df[target]


"""safe_loans_raw = df[df[target] == 1]
risky_loans_raw = df[df[target] == -1]
percentage = len(risky_loans_raw)/len(safe_loans_raw) #0.2327
safe_loans = safe_loans_raw.sample(math.ceil(percentage * safe_loans_raw.shape[0]))
risky_loans = risky_loans_raw
"""

for feature in features:
    one_hot_feature = pd.get_dummies(loans[feature])
    loans = loans.join(one_hot_feature)
    loans.drop(feature,axis = 1,inplace = True)
    
col = loans.columns
feature_encoded = []
for i in range(len(col)):
    feature_encoded.append(col[i])
    
train_data = loans.iloc[train_idx]
train_target = loans_target.iloc[train_idx]
train_ = train_data.join(train_target)
test_data = loans.iloc[test_idx]
test_target = loans_target.iloc[test_idx]


def intermediate_node_num_mistakes(labels_in_node):
    num_safe = sum(labels_in_node == 1)
    num_risky = sum(labels_in_node== -1)
    if num_safe > num_risky:
        mistakes = num_risky
    else:
        mistakes = num_safe
    return (mistakes)
    

"""# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print ('Test passed!')
else:
    print ('Test 1 failed... try again!')
    
# Test case 2
example_labels = np.array([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print ('Test passed!')
else:
    print ('Test 2 failed... try again!' )   
    
# Test case 3
example_labels = np.array([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print ('Test passed!')
else:
    print ('Test 3 failed... try again!')"""
    
    
def best_split_feature(data,features,label_name):
    #target_values = data[label_name]
    best_feature = None
    best_error = 1
    num_data_points = len(data)   
    for feature in features:
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        left_mistake = intermediate_node_num_mistakes(left_split[label_name])
        right_mistake = intermediate_node_num_mistakes(right_split[label_name])
        error = (left_mistake + right_mistake)/num_data_points
        
        if error < best_error:
            best_error = error
            best_feature = feature

    return (best_feature)
        
def create_leaf(target_values):
    leaf = {'splitting_feature' : None, 'left' : None, 'right' : None, 'is_leaf' : True}
    
    num_ones = np.sum(target_values == 1)
    num_minus_ones = np.sum(target_values == -1)
    
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1
    return (leaf)
    
def decision_tree_create(data,features,target,current_depth = 0,max_depth = 10):
    remaining_features = features[:]
    target_values = data[target]
    print("-------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth,len(target_values)))
    
    #stopping condition 1 : All data points in a node are from the same class.
    if intermediate_node_num_mistakes(data[target]) == 0:
        print ("Stopping condition 1 reached")
        return (create_leaf(target_values))
        
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print ("Stopping condition 2 reached.")
        return (create_leaf(target_values)) 
        
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  
        print ("Reached maximum depth. Stopping for now.")
        return (create_leaf(target_values))
        
    splitting_feature = best_split_feature(data,remaining_features,target)
    
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print ("Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split)))
                      
    if len(left_split) == len(data):
        print ("Creating leaf node.")
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print ("Creating leaf node.")
        return create_leaf(right_split[target])
        
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
   
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth) 

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

# my_tree = decision_tree_create(train_,feature_encoded,target,0,max_depth = 6)

def classify(tree,test_point,annotate = False):
    if tree['is_leaf']:
        if annotate:
           print ("At leaf, predicting %s" % (tree['prediction'] )) 
        return (tree['prediction'])
    else:
        split_feature_value = test_point[tree['splitting_feature']]
        if annotate:
            print ("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return (classify(tree['left'],test_point,annotate))
        else:
            return (classify(tree['right'],test_point,annotate))
# classify(my_tree,test_data.iloc[0,:],True)
            
def evaluate_classification_error(tree,data,target_data):
    correct = 0   
    total = data.shape[0]
    for i in range(data.shape[0]):
        actual=  target_data.iloc[i]
        prediction = classify(tree,data.iloc[i,:])
        correct = correct + (actual == prediction)
    return (1.0 * correct/total)
# evaluate_classification_error(my_tree,test_data,test_target)  0.62    
        
        