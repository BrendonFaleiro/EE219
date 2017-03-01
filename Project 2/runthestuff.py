import time
import numpy as np
import logging
import utils
import copy
import operator
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab as pl
import math

from collections import Counter
from collections import defaultdict
from pprint import pprint
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

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

def question_a():
    logging.info("<Question A> Plotting histogram")
    
    #dicts containing count of files of the given type
    train_count = {}
    test_count = {}
    
    for i in range(len(proc_train_set.target)):
        if train_set.target_names[train_set.target[i]] in train_count:
            train_count[train_set.target_names[train_set.target[i]]] +=1
        else:
            train_count[train_set.target_names[train_set.target[i]]] =1
    
    for i in range(len(test_set.target)):
        if test_set.target_names[test_set.target[i]] in test_count:
            test_count[test_set.target_names[test_set.target[i]]] +=1
        else:
            test_count[test_set.target_names[test_set.target[i]]] =1
    
    # plot histogram for number of documents vs. topic name
    pl.figure(1)
    pl.xlabel('Topic Name')
    pl.ylabel('Number of Topics')
    yloc = pl.arange(len(train_count.keys()))
    pl.title('Histogram of Number of Documents Per Topic')
    pl.yticks(yloc, train_count.keys())
    pl.barh(yloc, list(train_count.values()), align='center', color='green')
    pl.tight_layout()
    
    # get number of docs of each category
    CT_count_train = 0
    CT_count_test = 0
    RA_count_train = 0
    RA_count_test = 0
    
    for i in category_CT:
        CT_count_train += train_count[i]
        CT_count_test += test_count[i]
    
    for j in category_RA:
        RA_count_test += test_count[j]
        RA_count_train += train_count[j]
    
    logging.info('Computer Technology - train data: {0}'.format(CT_count_train))
    logging.info('Computer Technology - test data: {0}'.format(CT_count_test))
    logging.info('Recreational Activity - train data: {0}'.format(RA_count_train))
    logging.info('Recreational Activity - test data: {0}'.format(RA_count_test))
    
    pl.show()
    
def question_b():
    logging.info("<Question B> Getting the TFxIDF representation")
    utils.model_data(proc_train_set)
    
def question_c():
    logging.info("<Question C> Getting the significance and TFxICF representation")
    all_categories = train_full_set.target_names
    
    all_docs_per_category = []
    
    classes_list = [train_full_set.target_names.index("comp.sys.ibm.pc.hardware"), train_full_set.target_names.index("comp.sys.mac.hardware"), train_full_set.target_names.index("misc.forsale"), train_full_set.target_names.index("soc.religion.christian")]
    
    logging.info("Store data from all docs of a certain category as entries in all_data_category")
    for cat in all_categories:
        train_category = utils.fetch_data([cat])[0]
        data_category = train_category.data
        temp = ''
        for doc in data_category:
            temp += ' '+doc
        all_docs_per_category.append(temp)
        
    logging.info("Now build frequency tables for each class")
    
    vectorized_newsgroups_train  = utils.remove_stop_words(all_docs_per_category)
    
    print(vectorized_newsgroups_train.shape)
    
    max_term_freq_per_category = [0]*vectorized_newsgroups_train.shape[0]
    category_count_per_term = [0]*vectorized_newsgroups_train.shape[1]
    
    for i in range(vectorized_newsgroups_train.shape[0]):
        max_term_freq_per_category[i] = max(vectorized_newsgroups_train[i].data)
    
    category_count_per_term = vectorized_newsgroups_train.sum(axis=0)
    
    print(max_term_freq_per_category)
    print(category_count_per_term)
    
def question_d():
    logging.info("<Question D>Reducing data to 50 dimensional vector")
    train_idf= utils.model_data(proc_train_set)
    test_idf= utils.model_data(proc_test_set)
    _, _ = utils.apply_lsi(train_idf, test_idf)
    
def question_e():
    logging.info("<Question E> SVM Classification")
    clfy = svm.SVC(kernel='linear')
    utils.classify(clfy, "SVM", proc_train_set, proc_test_set,cv=False)
    
def question_f():
    logging.info("<Question F> SVM Classification with Cross Validation")
    clfy = svm.SVC(kernel='linear')
    utils.classify(clfy, "Cross validated SVM", proc_train_set, proc_test_set, cv=True)

def question_g():
    logging.info("<Question G> Bayes Classification")
    clfy = GaussianNB()
    utils.classify(clfy, "Bayes", proc_train_set, proc_test_set, cv=False)

def question_h():
    logging.info("<Question H> Logistic Regression")
    clfy = LogisticRegression(C=10)
    utils.classify(clfy, "Logistic Regression", proc_train_set, proc_test_set)

def question_i():
    categories = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey"]
    
    train, test = utils.fetch_data(categories)
    train.target = list(map(lambda x: int(0<=x and x< 4), train.target))
    test.target = list(map(lambda x: int(0<=x and x< 4), test.target))
    
    params = list(range(-3,4))
    l1_accuracies=[]
    l2_accuracies=[]
        
    for param in params:
        l1_classifier = LogisticRegression( penalty='l1', C=10**param, solver='liblinear')
        logging.info("Regularization Parameter set to {0}".format(param))
        l1_accuracies.append(utils.classify(l1_classifier, "Logistic Regression l1", train, test, cv=False, mean=True))
        l2_classifier = LogisticRegression( penalty='l2', C=10**param, solver='liblinear')
        l2_accuracies.append(utils.classify(l2_classifier, "Logistic Regression l2", train, test, cv=False, mean=True))
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(l1_accuracies)
    plt.xticks(range(6), [10 ** param for param in params])
    plt.title("Accuracy of L1 Logistic Regression vs regularization parameter")
    
    plt.subplot(212)
    plt.plot(l2_accuracies)
    plt.xticks(range(6), [10 ** param for param in params])
    plt.title("Accuracy of L2 Logistic Regression vs regularization parameter")
    plt.show()
    
def question_j():
    logging.info("<Question J> Multiclass Classification")
    category = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
    train, test = utils.fetch_data(category)

    
    train_idf = utils.model_data(train)
    test_idf = utils.model_data(test)
    logging.info("Creating TFxIDF Vector Representations")

    logging.info("Performing LSI on TFxIDF Matrices")
    # apply LSI to TDxIDF matrices
    svd = TruncatedSVD(n_components=50)
    train_lsi = svd.fit_transform(train_idf)
    test_lsi = svd.fit_transform(test_idf)

    logging.info("TFxIDF Matrices Transformed")

    logging.info("Size of Transformed Training Dataset: {0}".format(train_lsi.shape))
    logging.info("Size of Transformed Testing Dataset: {0}".format(test_lsi.shape))

    clf_list = [OneVsOneClassifier(GaussianNB()), OneVsOneClassifier(svm.SVC(kernel='linear')), OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(svm.SVC(kernel='linear'))]
    clf_name = ['OneVsOneClassifier Naive Bayes', 'OneVsOneClassifier SVM','OneVsRestClassifier Naive Bayes', 'OneVsRestClassifier SVM']

    # perform classification
    for clf,clf_n in zip(clf_list,clf_name):
        logging.info("Training {0} Classifier ".format(clf_n))
        clf.fit(train_lsi, train.target)
        logging.info("Testing {0} Classifier".format(clf_n))
        test_predicted = clf.predict(test_lsi)
        utils.calculate_stats(test.target, test_predicted)

    