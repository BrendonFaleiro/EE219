import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from scipy import linalg
from sklearn.cross_validation import KFold
from numpy import linalg as LA
from sklearn.metrics import auc, roc_curve

logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

def fetch_data():
    data = pd.read_csv('../Dataset/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # R_matrix is the ratings matrix.
    # Users on rows and movies on columns. 
    # Missing values are filled with 0.
    Ratings = data.pivot_table(index=['user_id'], columns=['movie_id'], values='rating', fill_value=0)
    Weights = Ratings.copy()
    Weights[Weights > 0] = 1
    return data.as_matrix(), Ratings.as_matrix(), Weights.as_matrix()

def calculate_error(predicted, actual, weights):
    error = actual - predicted
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(weights, squared_error)
    sum_squared_error = sum(sum(squared_error))

    return sum_squared_error
    
def nmf(Ratings, Weights, k, lambda_reg = 0):
    mask = np.sign(Ratings)
    eps = 1e-5
    rows, columns = Ratings.shape
    U = np.random.rand(rows, k)
    U = np.maximum(U, eps)

    V = linalg.lstsq(U, Ratings)[0]
    V = np.maximum(V, eps)

    masked_X = mask * Ratings

    for i in range(1, 100):

        top = np.dot(masked_X, V.T)
        bottom = (np.add(np.dot((mask * np.dot(U, V)), V.T), lambda_reg * U)) + eps
        U *= top / bottom
        U = np.maximum(U, eps)

        top = np.dot(U.T, masked_X)
        bottom = np.add(np.dot(U.T, mask * np.dot(U, V)), lambda_reg * V) + eps
        V *= top / bottom
        V = np.maximum(V, eps)

    return U,V
    
def factorize(Ratings, Weights, k):
    U, V = nmf(Ratings, Weights, k)
    R_mat_predicted = np.dot(U, V)
    sum_squared_error = calculate_error(R_mat_predicted, Ratings, Weights)
    return sum_squared_error
    #print(R_mat_predicted)
    
def find_MinSE(Ratings, Weights):
    features = (10, 50, 100)
    for k in features:
        squared_error = factorize(Ratings, Weights, k)
        logging.info('MSE of Rated Movies (k = {0}): {1}'.format(k, squared_error))
        
def cross_validate(data, Ratings):