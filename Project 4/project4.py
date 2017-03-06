import logging
import time
import argparse
import copy
import runthestuff

logging.basicConfig(level = logging.DEBUG, format='%(levelname)s %(asctime)s %(message)s')

def runCodeForQues(ques):
    logging.info("request for {0}".format(ques))
    if ques=='1':
        runthestuff.question_1()
    elif ques=='2':
        runthestuff.question_2()
    elif ques=='3':
        runthestuff.question_3()
    elif ques=='4':
        runthestuff.question_4()
    elif ques=='5':
        runthestuff.question_5()
    elif ques=='6':
        runthestuff.question_6()
    else:
        logging.info("Invalid question number")

def main():
    logging.info("Started main")
    parser = argparse.ArgumentParser(description = "execute question number (1,2,3,4,5,6)")
    parser.add_argument("--ques","-q")
    args = parser.parse_args()
    args_dict = vars(copy.deepcopy(args))
    ques = args_dict['ques']
    runCodeForQues(ques)

if __name__ == "__main__":
    main()
    