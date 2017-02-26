import utils
import logging

logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

def question_1():
    logging.info("Question 1: Matrix Factorization")
    
    data, Ratings, Weights = utils.fetch_data()
    utils.find_MinSE(Ratings, Weights)
    
def question_2():
    logging.info("Question 2: 10-fold Cross-Validation on Recommendation System")
    data, Ratings, Weight = utils.fetch_data()  
    utils.cross_validate(data, Ratings)  # perform cross validation
