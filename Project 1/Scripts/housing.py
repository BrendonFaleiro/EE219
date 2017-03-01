import utility
import pandas as pd
from sklearn import preprocessing


###########################################
# Question 4
# a. apply linear regression
# b. apply polynomial regression (with degree tuning)
###########################################

# LINEAR REGRESSION
def performLinearRegression(dataset):
    # obtain data for model creation
    Y = dataset['MEDV']  # class variable vector
    X = dataset.drop('MEDV', axis=1)  # feature vector

    utility.linearRegression(X,Y, "Housing", True)  # perform linear regression

# POLYNOMIAL REGRESSION
def performPolyRegression(dataset):
    # obtain data for model creation
    Y = dataset['MEDV']  # class variable vector
    X = dataset.drop('MEDV', axis = 1)  # feature vector

    utility.polynomialRegression(X,Y, "Housing")  # perform polynomial regression

###########################################
# Question 5
# a. perform ridge regression (tune alpha)
# b. perform lasso regularization (tune alpha)
###########################################

# RIDGE REGRESSION & LASSO REGULARIZATION
def overcomeOverfitting(dataset):
    # obtain data for model creation
    Y = dataset['MEDV']  # class variable vector
    X = dataset.drop('MEDV', axis = 1)  # feature vector

    utility.ridgeRegression(X,Y)  # perform ridge regression
    utility.lassoRegularization(X,Y)  # perform lasso regularization

    return True

def main(dataset, question):

    # to plot all the features vs. MEDV class variable
    # utility.plotFeatures(dataset, "Housing", "MEDV")

    if question == '4a':      performLinearRegression(dataset)
    elif question == '4b':    performPolyRegression(dataset)
    elif question == '5':     overcomeOverfitting(dataset)
    else:                     raise Exception("Housing dataset has only questions 4 and 5.")

if __name__ == "__main__":
    main()