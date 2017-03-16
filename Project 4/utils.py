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

warnings.simplefilter("ignore")
logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')
stop_words = text.ENGLISH_STOP_WORDS

def fetch_data(categories, single=True):
    '''
    Loads the 20_newsgroups datasets corresponding to the categories list
    if categories = [] then load everything
    '''
    if single:
        if categories ==[]:
            data = fetch_20newsgroups(subset='all', shuffle=True,random_state=42)
        else:
            data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True,random_state=42)
        return data
    else:
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
    

def get_vectorizer():
    return CountVectorizer(analyzer='word', stop_words=stop_words, ngram_range=(1, 1), tokenizer=StemTokenizer(), lowercase=True,max_df=0.99, min_df=2)
        
# Removes stop words
#First removes all words from the stopword list and keeps ony stems by ignoring plurals 
def model_data(data_set, name):
    logging.info("Preprocessing datasets")
    if os.path.isfile("./{}.pkl".format(name)):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./{}.pkl. Loading.".format(name))
        data = pickle.load(open("./{}.pkl".format(name), "rb"))
        return data
    else:
        stop_words = text.ENGLISH_STOP_WORDS
        vectorizer = get_vectorizer()
        train_counts = vectorizer.fit_transform(data_set.data)
        tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
        train_idf = tfidf_transformer.fit_transform(train_counts)
            
        logging.info("TFxIDF Matrices created")
        logging.info("Terms extracted: {0}".format(train_idf.shape[1]))
        
        pickle.dump( train_idf, open( "{0}.pkl".format(name), "wb" ) )
        return train_idf
    
def run_k_means(data_set):
    if os.path.isfile("./train_idf.pkl"):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./train_idf.pkl. Loading.")
        data = pickle.load(open("./train_idf.pkl", "rb"))
    else:
        data = model_data(data_set, "train_idf")
        
    labels = data_set.target//4
    print(labels)        
    predicted = KMeans(n_clusters=2, random_state=0).fit(data)
        
    calculate_stats(labels, predicted.labels_)

def calculate_stats(target, predicted):
    homogeneity = metrics.homogeneity_score(target, predicted)
    completeness = metrics.completeness_score(target, predicted)
    
    adjusted_Rand_Index = metrics.adjusted_rand_score(target, predicted)
    adjusted_Mutual_Info_Score = metrics.adjusted_mutual_info_score(target, predicted)
    
    logging.info("Accuracy: %0.3f" % metrics.accuracy_score(target, predicted))
    logging.info("Precision: %0.3f" % metrics.precision_score(target, predicted, average='macro'))
    logging.info("Recall: %0.3f" % metrics.recall_score(target, predicted, average='macro'))
    logging.info("Confusion Matrix: {0}".format(metrics.confusion_matrix(target, predicted)))
    logging.info("Homogeneity: %0.3f" % homogeneity)
    logging.info("Completeness: %0.3f" % completeness)
    logging.info("V-measure: %0.3f" % metrics.v_measure_score(target, predicted))
    logging.info("Adjusted Rand-Index: %.3f" % adjusted_Rand_Index)
    logging.info("Adjusted Mutual Info Score: %0.3f" % adjusted_Mutual_Info_Score)
    return (homogeneity, completeness, adjusted_Rand_Index, adjusted_Mutual_Info_Score)
    
def find_singular_values(data, k=50):
    U,S,V = scipy.sparse.linalg.svds(data, k)
    plt.plot(S[::-1])
    plt.ylabel('Singularities')
    plt.savefig('plots/singularities.png', format='png')
    plt.clf()
    
def non_linear_transformations(all, data_idf, reduced_dim):

    logging.info("Calculating for degree 2 polynomial..")
    svd = TruncatedSVD(n_components=reduced_dim)
    poly = PolynomialFeatures(degree=2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, poly, normalizer)

    X_lsa = lsa.fit_transform(data_idf)

    labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
    kmeans = KMeans(n_clusters=2).fit(X_lsa)

    calculate_stats(labels, kmeans.labels_)

    logging.info("Calculating for log features..")
    svd = TruncatedSVD(n_components=reduced_dim)
    poly = FunctionTransformer(np.log1p)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, poly, normalizer)

    X_lsa = lsa.fit_transform(data_idf)

    labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
    kmeans = KMeans(n_clusters=2).fit(X_lsa)

    calculate_stats(labels, kmeans.labels_)

def calculate_dim(data_set, name='train_idf'):
    if os.path.isfile("./{0}.pkl".format(name)):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./train_idf.pkl. Loading.")
        data = pickle.load(open("./{0}.pkl".format(name), "rb"))
    else:
        data = model_data(data_set,name)
    
    logging.info("Performing dimensionality reduction using LSA")
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    
    find_singular_values(data, k=100);
    ##################################################
    # from graph elbow is at 5
    maxh_svd=0
    dim_svd=0
    maxh_nmf=0
    dim_nmf=0
    arr= range(3,40)
    for n_comp in arr:
        svd = TruncatedSVD(n_comp)
        normalizer = Normalizer(norm='l2',copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(data)
        kmeans = KMeans(n_clusters=2).fit(X)
        logging.info("Stats for SVD dimension {}".format(n_comp))
        h,_,_,_=calculate_stats(data_set.target//4, kmeans.labels_)
        if maxh_svd<h:
            dim_svd = n_comp
            maxh_svd = h
    
    logging.info("Best dim for svd = {}".format(dim_svd))
    arr = range(3,40)
    for n_comp in arr:
        nmf_model = NMF(n_comp)    # NMF model, k=comp
        #normalizer = Normalizer(norm='l2', copy=True)
        lsa = make_pipeline(nmf_model)
        X = lsa.fit_transform(data)
        kmeans = KMeans(n_clusters=2).fit(X)
        logging.info("Stats for NMF dimension {}".format(n_comp))
        h,_,_,_ = calculate_stats(data_set.target//4, kmeans.labels_)
        if maxh_nmf<h:
            dim_nmf = n_comp
            maxh_nmf = h
    logging.info("Best dim for nmf = {}".format(dim_nmf))
    ####################################################################
    non_linear_transformations(data_set, data, dim_svd)
    
    labels = data_set.target//4
    svd = TruncatedSVD(n_components=2)
    reduced_tfidf = svd.fit_transform(data)
    reduced_tfidf_log = np.log1p(reduced_tfidf)
    x1 = reduced_tfidf_log[labels == 0][:, 0]
    y1 = reduced_tfidf_log[labels == 0][:, 1]
    x2 = reduced_tfidf_log[labels == 1][:, 0]
    y2 = reduced_tfidf_log[labels == 1][:, 1]
    plt.plot(x1,y1, 'r+')
    plt.plot(x2,y2,'g+')
    plt.savefig("plots/tf_idf_log.png", format='png')
    plt.clf()

def part4(data_set):
    if os.path.isfile("./train_idf.pkl"):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./train_idf.pkl. Loading.")
        data = pickle.load(open("./train_idf.pkl", "rb"))
    else:
        data = model_data(data_set,"train_idf")
    
    svd = TruncatedSVD(n_components=37)
    normalizer = Normalizer(norm='l2',copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(data)
    kmeans = KMeans(n_clusters=2).fit(X)
    svd2 = TruncatedSVD(n_components=2)
    X2 = svd2.fit_transform(X)
    
    x1 = X2[kmeans.labels_ == 0][:, 0]
    y1 = X2[kmeans.labels_ == 0][:, 1]
    print(x1)
    print(y1)
    plt.plot(x1,y1,'r+')
    x2 = X2[kmeans.labels_ == 1][:, 0]
    y2 = X2[kmeans.labels_ == 1][:, 1]
    print(x2)
    print(y2)
    plt.plot(x2, y2, 'g+')
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.savefig("plots/clusters_2d.png", format='png')
    plt.show()
 
def part5(data_set):
    labels = data_set.target
    if os.path.isfile("./full_train_idf.pkl"):  # load pickle file if it exists
        logging.info("Pre-processed Dataset located at ./train_idf.pkl. Loading.")
        data = pickle.load(open("./full_train_idf.pkl", "rb"))
    else:
        data = model_data(data_set,"full_train_idf")
    
    #Fix k
    k=20
    svd_metrics =[]
    nmf_metrics =[]
    metric_names = ['homogeneity_score', 'completeness_score', 'adjusted_rand_score', 'adjusted_mutual_info_score']
    ds=range(2,75)
    logging.info("Varying Dimensions with fixed k=20")
    for d in ds:
        logging.info("Dimension = {}".format(d))
        svd = TruncatedSVD(n_components=d)
        normalizer = Normalizer(copy=False)
        svd_pipeline = make_pipeline(svd, normalizer)
        
        X_SVD = svd_pipeline.fit_transform(data)
        kmeans_svd = KMeans(n_clusters=k).fit(X_SVD)
        svd_metrics.append(calculate_stats(labels, kmeans_svd.labels_))
        
        nmf = NMF(n_components=d)
        normalizer = Normalizer(copy=False)
        nmf_pipeline = make_pipeline(nmf,normalizer)
        X_NMF = nmf_pipeline.fit_transform(data)
        kmeans_nmf = KMeans(n_clusters=k).fit(X_NMF)
        nmf_metrics.append(calculate_stats(labels, kmeans_nmf.labels_))
    
    for i,metric_name in enumerate(metric_names):
        plt.plot(ds, list(map(lambda x: x[i], svd_metrics)), label = metric_name)
    plt.xlabel('Dimensions')
    plt.ylabel('Metric Value')
    plt.legend(loc='best')
    plt.savefig('plots/fixk_svd_metrics.png', format='png')
    plt.clf()
    
    for i,metric_name in enumerate(metric_names):
        plt.plot(range(2,31), map(lambda x: x[i], nmf_metrics), label = metric_name)
    plt.xlabel('Dimensions')
    plt.ylabel('Metric Value')
    plt.legend(loc='best')
    plt.savefig('plots/fixk_nmf_metrics.png', format='png')
    plt.clf()
    
    #Fix d and vary k
    s = find_singular_values(data, k=2000)
    
    
    svd_metrics=[]
    nmf_metrics=[]
    logging.info("Varying k for d=50")
    ks = range(4,31)
    for k in ks:
        logging.info("k = {0}".format(k))
        svd = TruncatedSVD(n_components=50)
        normalizer = Normalizer(copy = False)
        svd_pipeline = make_pipeline(svd,normalizer)
        X_SVD = svd_pipeline.fit_transform(data)
        kmeans_svd = KMeans(n_clusters).fit(X_SVD)
        svd_metrics.append(calculate_stats(labels, kmeans_svd.labels_))
        
        nmf = NMF(n_components = 50)
        normalizer = Normalizer(copy=False)
        nmf_pipeline = make_pipeline(nmf, normalizer)
        X_NMF = nmf_pipeline.fit_transform(data)
        kmeans_nmf = KMeans(n_clusters=k).fit(X_NMF)
        nmf_metrics.append(calculate_stats(labels, kmeans_svd.labels_))
        
    for i,metric_names in enumerate(metric_names):
        plt.plot(ks, list(map(lambda x: x[i], svd_metrics)), label = metric_names)
    plt.xlabel('Clusters')
    plt.ylabel('Metric Value')
    plt.legend(loc='best')
    plt.savefig('plots/fixd_svd_metrics.png', format='png')
    plt.clf()
    
    for i,metric_names in enumerate(metric_names):
        plt.plot(range(4,31), map(lambda x: x[i], nmf_metrics), label = metric_names)
    plt.xlabel('Clusters')
    plt.ylabel('Metric Value')
    plt.legend(loc='best')
    plt.savefig('plots/fixd_nmf_metrics.png', format='png')
    plt.clf() 
