#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:14:17 2024

@author: meghanapuli
"""

'''
Problem statement

Housing price prediction. 
The training data set contains many examples with 4 features (size, bedrooms, floors and age).
Note, the price is in 1000s dollars

We would like to build a linear regression model using these values so we can then predict the price 
for other houses - say, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.
'''

import numpy as np

# load the data set
def load_data():
    data = np.loadtxt("houses.txt", delimiter=',')
    #print(data.ndim)
    X = data[:,0:-1]
    y = data[:,-1]
    return X, y

# load the dataset
X_train, y_train = load_data()
#print(X_train)
#print(y_train)

m,n = X_train.shape

# predict the model output
def compute_model_output(X, w, b):
    f_wb = np.dot(X,w) + b
    return f_wb

# calculate the cost of the model
def compute_cost(X, y, w, b):
    cost = 0
    for i in range(m):
        f_wb = compute_model_output(X[i], w, b)
        cost += (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    return total_cost

# compute gradients
def compute_gradient(X, y, w, b): 
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = compute_model_output(X[i], w, b)
        error = f_wb - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i,j]
        dj_db = dj_db + error
    
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

# gradient descent to find optimal w,b
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w,b

# feature scaling
def zscore_normalize_features(X):

    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
    
# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
#print(X_norm)
#print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"\nPeak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"\nPeak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

initial_w =  np.zeros((n,))
initial_b = 0.
alpha = 1.0e-1     #try 1.0e-3
iterations = 1000  #try 10000

# run gradient descent 
w_final, b_final = gradient_descent(X_norm, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"\nFinal b,w found by gradient descent: {b_final:0.2f},{w_final}")

cost_of_our_model = compute_cost(X_norm, y_train, w_final, b_final)
print(f"\nCost of our model: {cost_of_our_model}")
print("\nHousing price prediction")
# test the model
X_house = []
X_house.append(float((input)("\nEnter the size of house(in sqft): ")))
X_house.append(float((input)("\nEnter the no. of bedrooms: ")))
X_house.append(float((input)("\nEnter the no. of floors:  ")))
X_house.append(float((input)("\nEnter the age of the house: ")))

# normalize the original features
x_house_norm = (X_house - X_mu) / X_sigma

esltimated_price = compute_model_output(x_house_norm, w_final, b_final)
print(f"Estimated price: ${round(esltimated_price)*1000}")
