#from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from supporting import make_DictionaryNL, extract_features, removeNoMeaningWords
from sklearn.metrics import confusion_matrix
import os, sys
from sklearn.metrics import accuracy_score

TOTALTRAINING=160
POSPTRAINING=80
TOTALTESTING=20
POSTESTING=10
full_path = os.path.realpath(__file__)
train_dir=os.path.dirname(full_path)+"\\\dataStudents\\training"
class_names=[0,1]
#weights={0:6, 1:1}
dictionary=make_DictionaryNL(train_dir)
wordCommonDic=removeNoMeaningWords(dictionary)
feature_names=[item[0] for item in wordCommonDic]

train_labels=np.zeros(TOTALTRAINING)
# negative reviews are labeled 0
train_labels[POSPTRAINING:TOTALTRAINING]=1 # positive reviews
train_matrix=np.load("features.npy")
print(train_matrix.size)
model=MultinomialNB()
model.fit(train_matrix,train_labels)
pred_result=model.predict(train_matrix)
print("neg:", pred_result[0:POSPTRAINING])
print("pos:", pred_result[POSPTRAINING:])
print ("confusion_matrix")
print(confusion_matrix(train_labels, pred_result))
p=accuracy_score(train_labels, pred_result)
print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(train_labels, pred_result))

#eval.
test_dir=os.path.dirname(full_path)+"\\dataStudents\\testing"
test_matrix= extract_features(test_dir, wordCommonDic)
print(test_matrix)
# # to compute the error rate
pred_result=model.predict(test_matrix)
print("Negative reviews in Testing:", pred_result[0:POSTESTING])
print("Positive reviews in Testing:", pred_result[POSTESTING:])
print ("confusion_matrix")
test_labels=np.zeros(TOTALTESTING)
test_labels[POSTESTING:]=1
print(confusion_matrix(test_labels, pred_result))
print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(test_labels, pred_result))
#print(model.predict(test_matrix[-3:]))