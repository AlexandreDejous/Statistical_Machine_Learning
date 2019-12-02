#note for the user : we assume that the data lives in the same folder as the program

#imports
from matplotlib import pyplot as p
from libsvm.svmutil import *
import numpy as np 

#read the L2 pre-computed kernel matrices corresponding to training, validation and testing data
yTrain, xTrain = svm_read_problem("trn_kernel_mat.svmlight")
yValid, xValid = svm_read_problem("val_kernel_mat.svmlight")
yTest, xTest = svm_read_problem("tst_kernel_mat.svmlight")

#define the C values
C = [0.01, 0.1, 1, 10, 100]
models = []

#for each value of C, train models on the matrix of the training data
for i in range(len(C)):
    string = "-c {}".format(C[i])
    models.append(svm_train(yTrain, xTrain, string))

#extract the number of support vectors from all the models 
nr_sv = []
for i in range(len(models)):
    nr_sv.append(models[i].get_nr_sv())

#establish predictions on the validation and training kernel matrices
predictTrain = []
predictValid = [] 

for i in range(len(models)):
    predictTrain.append(svm_predict(yTrain, xTrain, models[i]))
    predictValid.append(svm_predict(yValid, xValid, models[i]))
    

#extract the percentage of accuracy of these predictions, and convert it into a risk (error) by doing (1 - accuracy) = risk
riskTrain = []
riskValid = []

for i in range(len(models)):  
    riskTrain.append((100 - predictTrain[i][1][0])/100)
    riskValid.append((100 - predictValid[i][1][0])/100)

#plot the risk versus C
p.semilogx(C, riskTrain)
p.semilogx(C, riskValid)
p.xlabel("C (log scale)")
p.ylabel("Error")
p.legend(["Train. Err","Valid. Err"])



print("opening txt... \n")
#write informations in a table in a txt
txt = open("table_dejous.txt", "w")
print("writing in txt... \n")
string = "{0:^5s} {1:^11s} {2:^11s} {3:^9s}\n".format("C", "Train. Err", "Valid. Err", "Num of sv")
txt.write(string)
string = "----- ----------- ----------- ---------\n"
txt.write(string)
for i in range(len(C)):
    string = "{0:^5} {1:^11.3f} {2:^11.3f} {3:^9}\n".format(C[i], riskTrain[i], riskValid[i], nr_sv[i])
    txt.write(string)
print("finished writing ! \n")



#C = 100 minimizes the error on both sets. We thus choose it as the best regularization constant.
print("Best Constant C : 100 \n")

#For C = 100, the test error is as such :

predictTest100 = svm_predict(yTest, xTest, models[4])
riskTest100 = (100 - predictTest100[1][0])/100
print("Test Error for C = 100 : ", riskTest100,"\n")

#to find the minimal value of epsilon, we will use the equation to calculate the interval width presented during the second course.
#lmin and lmax are respectively the minimum and maximum of the loss function (0 and 1)
#l = number of samples
#gamma = level of confidence (0.99)
lmin = 0
lmax = 1
l = len(yTest)
epsilon = (lmax-lmin)*np.sqrt((np.log(2)-np.log(1-0.99))/(2*l))

print("The minimal value of epsilon is ", epsilon,"\n")

#show the graph
p.show()