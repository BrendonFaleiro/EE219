Team Members
Anurag Pande - 604749647
Brendon Faleiro - 704759007
Sachin Krishna Bhat - 304759727

CLASSIFICATION ANALYSIS
The following README contains the requirements and steps that are required to execute Project 2. 

IMPLEMENTATION
Dependencies:
a. numpy v1.10.4
b. scipy v0.16.1
c. matplotlib v1.5.1
d. sklearn v0.17
e. nlkt v3.0

Usage: python project2.py -q <question number>

<question number> : a, b, c, d, e, h, i, j
                     
For example, if you want to run Question a (Plot Histograms and Count of Documents),
$ cd Scripts 
$ python project2.py -q a

If you want to run Question h (Perform Logistic Regression),
$ cd Scripts
$ python project2.py -q h  

Description
- Import the entire zip file containing all the folders (Dataset, Graphs, Scripts) into an IDE (e.g PyCharm).
- Run project2.py with configurations as specified above.
- Graphs are displayed as a separate window.
- Output of each question is displayed in the console output.

Scripts : Contains all the python scripts required to execute the project.
          - project2.py : Main file that is executed to get results
          - runthestuff.py : Comprises for code for each question in the project.
          - utils.py : Imported by execute.py comprises of generic functions to execute