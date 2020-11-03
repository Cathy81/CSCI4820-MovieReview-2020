from sklearn.naive_bayes import BernoulliNB
import numpy as np
from supporting import make_DictionaryNL, extract_features, removeNoMeaningWords
from sklearn.metrics import confusion_matrix
import os, sys
from sklearn.metrics import accuracy_score

TOTALTRAINING=1800
TOTALTESTING=200
full_path = os.path.realpath(__file__)
train_dir=os.path.dirname(full_path)+"\\\data\\training"
class_names=[0,1]
#weights={0:6, 1:1}
dictionary=make_DictionaryNL(train_dir)
wordCommonDic=removeNoMeaningWords(dictionary)
feature_names=[item[0] for item in wordCommonDic]

train_labels=np.zeros(TOTALTRAINING)
# negative reviews are labeled 0
train_labels[900:1800]=1 # spam
train_matrix=np.load("features.npy")
print(train_matrix.size)
model=BernoulliNB()
model.fit(train_matrix,train_labels)
pred_result=model.predict(train_matrix)
print("neg:", pred_result[0:899])
print("pos:", pred_result[900:])
print ("confusion_matrix")
print(confusion_matrix(train_labels, pred_result))
p=accuracy_score(train_labels, pred_result)
print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(train_labels, pred_result))

#eval.
test_dir=os.path.dirname(full_path)+"\\data\\testing"
test_labels=np.zeros(TOTALTESTING)
test_labels[100:]=1
test_matrix= extract_features(test_dir, wordCommonDic)
# # to compute the error rate
pred_result=model.predict(test_matrix)
print("Negative reviews in Testing:", pred_result[0:99])
print("Positive reviews in Testing:", pred_result[100:])
print ("confusion_matrix")
print(confusion_matrix(test_labels, pred_result))
print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(test_labels, pred_result))
