import logging
import time
import argparse
import copy
import runthestuff


logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

def runCodeForQues(ques):
    logging.info("request for {0}".format(ques))
    if ques=='a':
        runthestuff.question_a()
    elif ques=='b':
        runthestuff.question_b()
    elif ques=='c':
        runthestuff.question_c()
    elif ques=='d':
        runthestuff.question_d()
    elif ques=='e':
        runthestuff.question_e()
    elif ques=='f':
        runthestuff.question_f()
    elif ques=='g':
        runthestuff.question_g()
    elif ques=='h':
        runthestuff.question_h()
    elif ques=='i':
        runthestuff.question_i()
    elif ques=='j':
        runthestuff.question_j()
    else:
        logging.info("Invalid question number")

def main():
    logging.info("Started main")
    parser = argparse.ArgumentParser(description = "execute question number (a,b,c,d,e,f,g,h,i,j)")
    parser.add_argument("--ques","-q")
    args = parser.parse_args()
    args_dict = vars(copy.deepcopy(args))
    ques = args_dict['ques']
    runCodeForQues(ques)
    



if __name__ == "__main__":
    main()
    