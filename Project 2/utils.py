import logging
import re
import string
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab as pl
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import sklearn.metrics as smet
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')


logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

def fetch_data(categories):
    '''
    Loads the 20_newsgroups datasets corresponding to the categories list
    if categories = [] then load everything
    '''
    if categories ==[]:
        train = fetch_20newsgroups(subset='train', shuffle=True,random_state=42)
        test = fetch_20newsgroups(subset='test', shuffle=True,random_state=42)
    else:
        train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,random_state=42)
        test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True,random_state=42)
    return train,test


class StemTokenizer(object):
    def __init__(self):
        self.snowball_stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.regex_tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        tmp = [self.snowball_stemmer.stem(t) for t in self.regex_tokenizer.tokenize(doc)]
        return tmp
# Removes stop words
#First removes all words from the stopword list and keeps ony stems by ignoring plurals 
def remove_stop_words(data):
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, ngram_range=(1, 1), tokenizer=StemTokenizer(), lowercase=True,max_df=0.99, min_df=2)
    train_counts = vectorizer.fit_transform(data)
    return train_counts
    
def model_data(data_set):
    logging.info("Preprocessing datasets")
        
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, ngram_range=(1, 1), tokenizer=StemTokenizer(), lowercase=True,max_df=0.99, min_df=2)
    train_counts = vectorizer.fit_transform(data_set.data)
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    train_idf = tfidf_transformer.fit_transform(train_counts)
        
    logging.info("TFxIDF Matrices created")
    logging.info("Terms extracted: {0}".format(train_idf.shape[1]))
    
    return train_idf

def apply_lsi(train_set, test_set):
    logging.info("Applying LSI")
    svd = TruncatedSVD(n_components=50, random_state=42)
    train_lsi = svd.fit_transform(train_set)
    test_lsi = svd.fit_transform(test_set)
    
    logging.info("TFxIDF Transformed")
    logging.info("Size of training data set: {0}".format(train_lsi.shape))
    logging.info("Size of testing data set: {0}".format(test_lsi.shape))
    return train_lsi, test_lsi

def classify(classifier, algo, train_set, test_set, cv=False, roc=True, mean=False):
    train_idf = model_data(train_set)
    test_idf = model_data(test_set)
    train_lsi, test_lsi = apply_lsi(train_idf, test_idf)
    
    if cv:
        logging.info("Calculating best C parameter")
        C = [-2,-1,0,1,2,3]
        best_scores = []
        
        for c in C:
            logging.info("C = {}".format(c))
            clf = svm.SVC(kernel='linear', C = 10**c)
            scores = cross_validation.cross_val_score(clf, train_lsi, train_set.target,cv=5)
            best_scores.append(np.mean(scores))
        best_index = best_scores.index(max(best_scores))
        logging.info("Using parameter:{}".format(C[best_index]))
        classifier = svm.SVC(kernel='linear', C = 10**best_index)
        
    logging.info("Training {0} classifier.".format(algo))
    classifier.fit(train_lsi, train_set.target)
    logging.info("Testing {0} classifier.".format(algo))
    test_predicted = classifier.predict(test_lsi)
    
    accuracy =calculate_stats(test_set.target, test_predicted)
    if mean:
        logging.info("Mean: {0}".format(np.mean(classifier.coef_)))
    if roc:
        plotROC(test_set.target, test_predicted, algo)
    
    return accuracy
    
def calculate_stats(target, predicted):
    accuracy = smet.accuracy_score(target, predicted)
    precision = smet.precision_score(target, predicted, average='macro')
    recall = smet.recall_score(target, predicted, average='macro')
    confusion = smet.confusion_matrix(target, predicted)
    logging.info("Stats:")
    logging.info("Accuracy: {0}".format(accuracy))
    logging.info("Precision: {0}".format(precision))
    logging.info("Recall: {0}".format(recall))
    logging.info("Confusion Matrix: {0}".format(confusion))
    return accuracy
  
def plotROC(target, predicted, algo):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr,tpr,_ = roc_curve(target, predicted)
    roc_auc = auc(fpr,tpr)
    
    pl.figure(1)
    pl.plot(fpr,tpr, label='ROC curve(area = {0:0.4f})'.format(roc_auc))
    pl.plot([0,1], [0,1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curves for {0} Classifier'.format(algo))
    pl.legend(loc="lower right")
    pl.show()
    

  
def main():
    remove_stop_words("H!_@i 1_23 !! @; \[] Yessir")
    
if __name__ =="__main__" :
    main()