import logging
import numpy as np
import re
import pickle
import os
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

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
    
    pickle.dump( train_idf, open( "train_idf.pkl", "wb" ) )
    return train_idf
    
def run_k_means(data_set):
    if os.path.isfile("./train_idf.pkl"):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./train_idf.pkl. Loading.")
        data = pickle.load(open("./train_idf.pkl", "rb"))
    else:
        data = model_data(data_set)
        
    labels = data_set.target
    true_k = np.unique(labels).shape[0]
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,verbose=False)
        
    predicted = km.fit(data)
        
    calculate_stats(labels, km.labels_, data)

def calculate_stats(target, predicted, data):
    logging.info("Accuracy: %0.3f" % metrics.accuracy_score(target, predicted))
    logging.info("Precision: %0.3f" % metrics.precision_score(target, predicted, average='binary'))
    logging.info("Recall: %0.3f" % metrics.recall_score(target, predicted, average='binary'))
    logging.info("Confusion Matrix: {0}".format(metrics.confusion_matrix(target, predicted)))
    logging.info("Homogeneity: %0.3f" % metrics.homogeneity_score(target, predicted))
    logging.info("Completeness: %0.3f" % metrics.completeness_score(target, predicted))
    logging.info("V-measure: %0.3f" % metrics.v_measure_score(target, predicted))
    logging.info("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(target, predicted))
    logging.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, predicted, sample_size=1000))
    logging.info("Adjusted Mutual Info Score: %0.3f" % metrics.adjusted_mutual_info_score(target, predicted))
    
def calculate_dim(data_set):
    if os.path.isfile("./train_idf.pkl"):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./train_idf.pkl. Loading.")
        data = pickle.load(open("./train_idf.pkl", "rb"))
    else:
        data = model_data(data_set)
    print("Performing dimensionality reduction using LSA")
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    exp_variance = []
    k=0;
    for n_comp in np.arange(10,2000,100):
        svd = TruncatedSVD(n_comp)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(data)

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step for {0} comp: {1}%".format(n_comp, int(explained_variance * 100)))
        
    