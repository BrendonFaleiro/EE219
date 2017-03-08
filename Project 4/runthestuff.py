import logging
import numpy as np
import utils
import copy

logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

category_CT = categories[:4]
category_RA = categories[4:]

train_set, test_set = utils.fetch_data(categories)

proc_train_set = copy.deepcopy(train_set)
proc_test_set = copy.deepcopy(test_set)

for i,j in enumerate(proc_train_set.target):
    if j>=0 and j<4:
        proc_train_set.target[i] = 0
    else:
        proc_train_set.target[i] = 1

proc_train_set.target_name = ['Computer Technology', 'Recreational Activity']
        
for i,j in enumerate(proc_test_set.target):
    if 0<=j and j<4:
        proc_test_set.target[i] = 0
    else:
        proc_test_set.target[i] = 1

proc_test_set.target_names = ['Computer Technology', 'Recreational Activity']

train_full_set, test_full_set = utils.fetch_data([])

def question_1():
    logging.info("<Question 1> Getting the TFxIDF representation")
    utils.model_data(proc_train_set)
    
def question_2():
    logging.info("<Question 2> Performing k-means clustering")
    utils.run_k_means(proc_train_set)
    
def question_3():
    logging.info("<Question 3> Reducing dimensionality")
    utils.calculate_dim(proc_train_set)
    