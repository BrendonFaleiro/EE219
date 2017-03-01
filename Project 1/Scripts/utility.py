import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation, linear_model
from sklearn.preprocessing import PolynomialFeatures

# PLOT RELATIONSHIP BETWEEN CLASS AND ATTRIBUTES
def plotFeatures(dataset, datasetName, className):
    for attribute in dataset.columns:
        if attribute == className:
            continue

        X = dataset[attribute]
        Y = dataset[className]

        plt.scatter(X, Y)
        plt.title('Prediction Plot')
        plt.xlabel(attribute)
        plt.ylabel(className)

        plt.savefig('../Graphs/{0}/Attribute Relationship/{1} vs {2}.png'.format(datasetName, className, attribute))

        plt.show()


# LINEAR REGRESSION
def linearRegression(X,Y, datasetName, workFlow=False, workFlowNo=0):
    print ("LINEAR REGRESSION")
    print ("Executing...")

    # can change to model on the entire dataset but by convention splitting the dataset is a better option
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)

    lm = linear_model.LinearRegression()  # sklearn Linear Regression model used for cross validation
    lm.fit(X_train,Y_train)

    scores = cross_validation.cross_val_score(lm, X, Y, cv=10, scoring='mean_squared_error')  # cross validation 10 folds
    predicted = cross_validation.cross_val_predict(lm, X, Y, cv=10)

    if workFlow:
        est = sm.OLS(Y, X).fit()  # panda OLS library used to build model on entire dataset and provide stats on variable
        print (est.summary())
    else:
        print ("WORKFLOW_" + str(workFlowNo))
        print ("Estimated intercept coefficient: ", lm.intercept_)
        data = list(zip(X.columns, lm.coef_))
        estimatedCoefficients = pd.DataFrame(data, columns=['Features', 'EstimatedCoefficients'])
        print (estimatedCoefficients)

    rmseEstimator = np.mean(scores*-1)**0.5  # average of the RMSE values

    print()
    print ("RMSE Values of Estimator : " + str(rmseEstimator))
    print()

    # plot graph for Fitted values vs Actual values
    plt.scatter(Y, predicted)
    plt.figure(1)
    plt.xlabel("Actual Median Value")
    plt.ylabel("Predicted Median Value")
    plt.title('Fitted values vs Actual Values')
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)

    if workFlow:
        plt.savefig("../Graphs/{0}/Question {1}/LR - Fitted vs Actual.png".format(datasetName,
                                                                                  '2a' if datasetName == "Network" else '4a'))
    else:
        plt.savefig("../Graphs/{0}/Question 3/LR - Fitted vs Actual WorkFlow {1}.png".format(datasetName,workFlowNo))

    # plot graph for Residual values vs Fitted Values
    plt.figure(2)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values plot')
    plt.scatter(predicted, predicted - Y, c='b', s=40, alpha=0.5)
    plt.hlines(y=0, xmin=-10, xmax=10)

    if workFlow:
        plt.savefig("../Graphs/{0}/Question {1}/LR - Residuals vs Fitted.png".format(datasetName,
                                                                                    '2a' if datasetName == "Network" else '4a'))
    else:
        plt.savefig("../Graphs/{0}/Question 3/LR - Residuals vs Fitted WorkFlow {1}.png".format(datasetName, workFlowNo))

    plt.show()

# RANDOM FOREST REGRESSION
def randomForestRegression(X,Y,datasetName):
    print ("RANDOM FOREST REGRESSION")
    print ("Executing...")
    print()

    # fine tuning of depth of tree
    depth = range(4,15)
    rmse = []

    for eachDepth in range(len(depth)):  # test depth of tree using 10 folds cross validation
        estimator = RandomForestRegressor(n_estimators=20, max_depth=depth[eachDepth])
        scores = cross_validation.cross_val_score(estimator, X, Y, cv=10, scoring='mean_squared_error')
        rmse.append((np.mean(scores)*-1)**(0.5))

    # plot graphs for RMSE vs Depth of Tree
    plt.figure(1)
    plt.plot(depth,rmse)
    plt.title('RMSE vs Depth of Tree')
    plt.xlabel('Depth')
    plt.ylabel('Root Mean Square Error')
    plt.savefig("../Graphs/{0}/Question 2b/RMSE vs Depth of Tree.png".format(datasetName))

    bestDepth = depth[rmse.index(min(rmse))]  # best depth obtained on basis of RMSE
    # fine tuning of number of maximum tree
    noOfTrees = range(20,220,20)
    rmse = []

    for eachTreeSize in range(len(noOfTrees)):  # test maximum number of tree using 10 folds cross validation
        estimator = RandomForestRegressor(n_estimators=noOfTrees[eachTreeSize], max_depth=bestDepth)
        scores = cross_validation.cross_val_score(estimator, X, Y, cv=10, scoring='mean_squared_error')
        rmse.append((np.mean(scores)*-1)**(0.5))

    bestTrees = noOfTrees[rmse.index(min(rmse))]

    # plot graphs for RMSE vs Maximum No. of Tree

    plt.figure(2)
    plt.plot(noOfTrees,rmse)
    plt.title('RMSE vs Maximum No of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Root Mean Square Error')
    plt.savefig("../Graphs/{0}/Question 2b/RMSE vs No of Trees.png".format(datasetName))

    rfe = RandomForestRegressor(n_estimators=bestTrees, max_depth=bestDepth)
    predicted = cross_validation.cross_val_predict(rfe, X, Y, cv=10)

    # plot graph for Fitted values vs Actual values
    plt.figure(3)
    plt.scatter(Y, predicted)
    plt.xlabel("Actual Median Value")
    plt.ylabel("Predicted Median Value")
    plt.title('Fitted values vs Actual Values')
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    plt.savefig("../Graphs/{0}/Question {1}/Fitted vs Actual.png".format(datasetName,
                                                                                  '2b' if datasetName == "Network" else '4a'))

    plt.figure(4)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values plot')
    plt.scatter(predicted, predicted - Y, c='b', s=40, alpha=0.5)
    plt.hlines(y=0, xmin=-10, xmax=10)

    plt.savefig("../Graphs/{0}/Question {1}/Residuals vs Fitted.png".format(datasetName,
                                                                                    '2b' if datasetName == "Network" else '4a'))

    # print the best rmse of random forest with maximum no of trees = 140 and depth = 9
    print ("RANDOM FOREST REGRESSION")
    print ("PARAMETER TUNING")
    print ("Max Depth : " + str(bestDepth))
    print ("Number of Maximum Tree " + str(bestTrees))
    print ("Root Mean Squared Error : " + str(min(rmse)))

    plt.show()


# NEURAL NETWORK REGRESSION
def neuralNetworkRegression(X,Y):
    print ("NEURAL NETWORK REGRESSION")
    print ("Executing...")

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)
    Y_test = Y_test.reshape(-1,1)
    Y_train = Y_train.reshape(-1,1)
    RMSEerror = []

    train = np.vstack((X_train, X_test))  # append both testing and training into one array
    outputTrain = np.vstack((Y_train, Y_test))
    outputTrain = outputTrain.reshape( -1, 1 )

    inputSize = train.shape[1]
    targetSize = outputTrain.shape[1]

    ds = SupervisedDataSet(inputSize, targetSize)
    ds.setField('input', train)
    ds.setField('target', outputTrain)

    hiddenSize = 100
    epochs = 100  # got after parameter tuning

    # neural network training model
    net = buildNetwork( inputSize, hiddenSize, targetSize, bias = True )
    trainer = BackpropTrainer(net, ds)

    # uncomment out to plot epoch vs rmse
    # takes time to execute as gets best epoch value
    # getting the best value of epochs

    print ("training for {} epochs...".format( epochs ))
    '''
    for i in range(epochs):
        print (i)
        mse = trainer.train()
        rmse = mse ** 0.5
        RMSEerror.append(rmse)

    plt.plot(range(epochs), RMSEerror)
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Epochs")
    plt.savefig("../Graphs/Network/Question 2c/RMSE vs Epochs.png")

    plt.show()
    '''
    print ("Model training in process...")
    train_mse, validation_mse = trainer.trainUntilConvergence(verbose = True, validationProportion = 0.15, maxEpochs = epochs, continueEpochs = 10)
    p = net.activateOnDataset(ds)
    
    mse = mean_squared_error(outputTrain, p)
    rmse = mse ** 0.5

    print ("Root Mean Squared Error for Best Parameters : " + str(rmse))

# POLYNOMIAL REGRESSION
def polynomialRegression(X,Y, datasetName):
    print ("POLYNOMIAL REGRESSION")
    print ("Executing...")

    # can change to model on the entire dataset but by convention splitting the dataset is a better option
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)

    # get the best degree of freedom
    degrees = range(1,15)
    rmse = []
    print ("Testing for the best degree...")

    for deg in range(len(degrees)):
        polyFeatures = PolynomialFeatures(degree= degrees[deg],interaction_only=True, include_bias= False)
        lm = linear_model.LinearRegression()

        pipeline = Pipeline([("polynomial_features", polyFeatures),
                             ("linear_regression", lm)])

        pipeline.fit(X,Y)
        scores = cross_validation.cross_val_score(pipeline, X, Y, scoring="mean_squared_error", cv=10)

        rmse.append(np.mean(abs(scores))**0.5)

    # plot graph for RMSE vs. degree of polynomial
    plt.figure(1)
    plt.plot(degrees,rmse)
    plt.ylabel("RMSE")
    plt.xlabel("Polynomial Degree")
    plt.title("RMSE vs Polynomial Degree")

    # best degree = housing - 3
    # best degree = network - 5

    degrees = [degrees[rmse.index(min(rmse))]]
    rmse = []

    # apply transformation on dataset to the best degree of freedom and then find error
    for deg in range(len(degrees)):
        polyFeatures = PolynomialFeatures(degree= degrees[deg],interaction_only=True, include_bias= False)
        X_train_trans = polyFeatures.fit_transform(X_train)
        X_test_trans = polyFeatures.fit_transform(X_test)

        lm = linear_model.LinearRegression()
        lm.fit(X_train_trans,Y_train)
        predicted = lm.predict(X_test_trans)

        rmse.append(np.mean(abs(predicted-Y_test)**0.5))

    plt.figure(1)
    plt.scatter(degrees,rmse)
    plt.ylabel("RMSE")
    plt.xlabel("Polynomial Degree")
    plt.title("RMSE vs Polynomial Degree")

    print ("Best Degree of Polynomial : " + str(degrees[0]))
    print ("Root Mean Squared Error for Best Parameters : " + str(rmse[0]))

    plt.savefig("../Graphs/{0}/Question {1}/RMSE vs Poly. Degree.png".format(datasetName,
                                                                             '3' if datasetName == "Network" else '4b'))

    plt.show()

# RIDGE REGRESSION
def ridgeRegression(X,Y):
    tuningAlpha = [1,0.1,0.01,0.001]

   # can change to model on the entire dataset but by convention splitting the dataset is a better option
   # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)

    ridge = RidgeCV(normalize=True,scoring='mean_squared_error', alphas=tuningAlpha, cv=10)
    ridge.fit(X, Y)
    prediction = ridge.predict(X)

    print ("RIDGE REGRESSION")
    print ("Best Alpha value for Ridge Regression : " + str(ridge.alpha_))
    print ('RMSE for corresponding Alpha =', np.sqrt(mean_squared_error(Y, prediction)))


# LASSO REGULARIZATION
def lassoRegularization(X,Y):
    tuningAlpha = [1,0.1,0.01,0.001]
    lasso = LassoCV(normalize=True, alphas=tuningAlpha, cv=10)
    lasso.fit(X,Y)
    prediction = lasso.predict(X)

    print ("LASSO REGULARIZATION")
    print ("Best Alpha value for Lasso Regularization : " + str(lasso.alpha_))
    print ('RMSE for corresponding Alpha =', np.sqrt(mean_squared_error(Y, prediction)))
