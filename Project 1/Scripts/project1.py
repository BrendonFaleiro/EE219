import csv
import copy
import network
import housing
import argparse
import pandas as pd

def readDataset(datasetFile, datasetName):
    """
    :param datasetPath: filename of the dataset
    :return: readData numpy array comprising of the dataset as string
    """
    offset = 1 if datasetName.lower() == "network" else 0

    # read data from csv files
    with open(datasetFile, 'r') as f:
        reader = csv.reader(f)
        readData = pd.DataFrame(list(reader)[offset:])

    # assign column names to corresponding dataset
    if datasetName.lower() == "network":
        readData.columns = ['weekIndex', 'weekDay', 'backupStartTime', 'workFlow', 'fileName', 'backUpSize', 'backUpTime']

        readData['weekIndex'] = readData['weekIndex'].astype(int)
        readData['backUpSize'] = readData['backUpSize'].astype(float)  # change to accommodate big decimal values

    else:
        # write code to add columns for housing data
        readData.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV" ]
        readData = readData.astype(float)

    return readData


if __name__ == "__main__":
    # add parser arguments to enable question wise execution
    parser = argparse.ArgumentParser()
    parser.add_argument("--ques", "-q", help="execute question number (Network - 1, 2a, 2b, 2c, 3 ; Housing - 4a, 4b, 5)")
    args = parser.parse_args()
    argsDict = vars(copy.deepcopy(args))

    questionNo = argsDict['ques']
    datasetName = "housing" if questionNo == '4a' or questionNo == '4b' or questionNo == '5' else "network"

    dataset = readDataset('../Dataset/network_backup_dataset.csv', datasetName) if datasetName == 'network' \
                                                                                else readDataset('../Dataset/housing_data.csv', datasetName)

    if datasetName.lower() == 'network':    network.main(dataset, questionNo)
    else:                                   housing.main(dataset, questionNo)