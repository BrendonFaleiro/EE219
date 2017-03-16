import logging
import numpy as np
import utils
import part5
import part6
import copy
import logging
import numpy as np
import re
import pickle
import warnings
import os
import matplotlib.pyplot as plt
import scipy
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures, FunctionTransformer

logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

computer_technologies = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x'
]

recreational_activity = [
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey'
]

science = [
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space'
]

miscellaneus = [ 'misc.forsale' ]

politics = [
    'talk.politics.misc',
    'talk.politics.guns',
    'talk.politics.mideast'
]

religion = [
    'talk.religion.misc',
    'alt.atheism',
    'soc.religion.christian'
]

classes = [computer_technologies, recreational_activity, science, miscellaneus, politics, religion]

category_CT = categories[:4]
category_RA = categories[4:]

data_set = utils.fetch_data(categories)
full_data_set = utils.fetch_data([])

def question_1():
    logging.info("<Question 1> Getting the TFxIDF representation")
    utils.model_data(data_set,"train_idf")
    
def question_2():
    logging.info("<Question 2> Performing k-means clustering")
    utils.run_k_means(data_set)
    
def question_3():
    logging.info("<Question 3> Reducing dimensionality")
    utils.calculate_dim(data_set)
    
def question_4():
    logging.info("<Question 4>")
    utils.part4(data_set)

def question_5():
    logging.info("<Question 5>")
    utils.part5(data_set)

def question_6():
    logging.info("<Question 6>")
    part6.main()
    