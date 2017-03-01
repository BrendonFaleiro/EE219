import utility
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

warnings.filterwarnings("ignore")  # ignore all warnings

###########################################
# Question 1
# plot relationship between copy size over
# time period of 20 days
# graphs displayed & stored in Graphs folder
###########################################

def plotRelationships(dataset):
    """
    :param dataset: network panda dataframe
    :return: plots of each workflows representing relationship between copy size & time
    """
    print ("PLOT RELATIONSHIPS")
    print ("Executing...")
   
    grouped = dataset.groupby(['weekIndex', 'weekDay','workFlow','fileName']).backUpSize.aggregate(np.sum).reset_index()

    for eachWorkFlow in sorted(pd.unique(dataset.workFlow.ravel())):
        workFlowData = grouped.loc[grouped['workFlow'] == eachWorkFlow]

        # change weekIndex and weekDay into dayNumber to plot over a time period of 20 days
        workFlowData['weekIndex'] = workFlowData['weekIndex'].apply(lambda x: 7 * (x-1))
        workFlowData = pd.DataFrame({'dayNumber' : workFlowData['weekIndex'] + workFlowData['weekDay'], 'fileName': workFlowData['fileName'],'backUpSize' : workFlowData['backUpSize']})
        workFlowData = workFlowData[workFlowData['dayNumber'] < 20]

        fig, ax = plt.subplots()
        labels = [] # file names stored in list

        for key, grp in workFlowData.groupby(['fileName']):
            ax = grp.plot(ax=ax, kind='line', x='dayNumber', y='backUpSize')
            labels.append("File_" + str(key))

        lines, _ = ax.get_legend_handles_labels()

        # assign x/y labels to the figure
        plt.xlabel('Time Period', fontsize=18)
        plt.ylabel('Copy Size (GBs)', fontsize=16)

        # assign legends & store figures on disk
        ax.legend(lines, labels, loc='best')
        ax.set_title('WorkFlow_' + str(eachWorkFlow))

        fig.savefig('../Graphs/Network/Question 1/WorkFlow_{0}.png'.format(eachWorkFlow))

    print ("Executed.")
    plt.show()  # display the graphs

###########################################
# Question 2
# a. apply linear regression
# b. apply random forest regression (with parameter tuning)
# c. apply neural network regression (with parameter tuning)
###########################################

# NORMALIZE DATA
def normalizeDataset(dataset):
    minMaxScaler = preprocessing.MinMaxScaler()
    xScaled = minMaxScaler.fit_transform(dataset)
    xNormalized = pd.DataFrame(xScaled)

    return xNormalized

# LINEAR REGRESSION
def performLinearRegression(dataset):
    # obtain data for model creation
    Y = dataset['backUpSize']  # class variable vector
    X = dataset.drop('backUpSize', axis = 1)  # feature vector
    X = normalizeDataset(X)  # normalize data

    utility.linearRegression(X,Y,"Network", workFlow=True) # perform linear regression

# RANDOM FOREST
def performRandomForest(dataset):
    # obtain data for model creation
    Y = dataset['backUpSize']  # class variable vector
    X = dataset.drop('backUpSize', axis = 1)  # feature vector
    X = normalizeDataset(X)  # normalize data

    utility.randomForestRegression(X,Y,"Network")  # perform random forest regression

# NEURAL NETWORKS
def performNeuralRegression(dataset):
    # obtain data for model creation
    Y = dataset['backUpSize']  # class variable vector
    X = dataset.drop('backUpSize', axis = 1)  # feature vector
    X = normalizeDataset(X)  # normalize data

    utility.neuralNetworkRegression(X,Y)  # perform neural network regression


###########################################
# Question 3
# a. predict backup size of each workflow separately
# b. apply polynomial regression
###########################################

# POLYNOMIAL REGRESSION
def performPolyRegression(dataset):
    for eachWorkFlow in sorted(pd.unique(dataset.workFlow.ravel())):
        workFlowData = dataset.loc[dataset['workFlow'] == eachWorkFlow]
        workFlowData = workFlowData.drop('workFlow', axis=1)

        # obtain data for model creation
        Y = workFlowData['backUpSize']  # class variable vector
        X = workFlowData.drop('backUpSize', axis = 1)  # feature vector
        X = normalizeDataset(X)  # normalize data

        utility.linearRegression(X,Y, "Network", False, eachWorkFlow)  # perform linear regression

    # obtain data for model creation
    Y = dataset['backUpSize']  # class variable vector
    X = dataset.drop('backUpSize', axis=1)  # feature vector
    X = normalizeDataset(X)  # normalize data

    utility.polynomialRegression(X,Y, "Network")
    


# PREPROCESS DATASET
def preprocessDataset(networkDataset):
    # enumerate weekDays, workFlow and fileName to integer type of data

    networkDataset = networkDataset.replace({'weekDay': {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2 , 'Thursday' : 3, 'Friday' : 4,
                                                     'Saturday' : 5, 'Sunday' : 6 }})

    uniqueWorkFlow = sorted(pd.unique(networkDataset['workFlow']))  # get unique workFlow values
    uniqueFiles = ['File_{0}'.format(s) for s in range(len((pd.unique(networkDataset['fileName']))))]   # get unique fileName values

    for i,j in zip(uniqueWorkFlow,range(len(uniqueWorkFlow))):
        networkDataset = networkDataset.replace({'workFlow': {i : j}})

    for i,j in zip(uniqueFiles,range(len(uniqueFiles))):
        networkDataset = networkDataset.replace({'fileName': {i : j}})

    return networkDataset


def main(dataset, question):
    dataset = preprocessDataset(dataset)  # preprocess the dataset to enable operations

    if question == '1':       plotRelationships(dataset)        # question 1.
    elif question == '2a':    performLinearRegression(dataset)  # question 2a.
    elif question == '2b':    performRandomForest(dataset)      # question 2b.
    elif question == '2c':    performNeuralRegression(dataset)  # question 2c.
    elif question == '3':     performPolyRegression(dataset)    # question 3.
    else:                     raise Exception("Network dataset has only questions 1, 2a, 2b, 2c and 3.")

if __name__ == "__main__":
    main()
