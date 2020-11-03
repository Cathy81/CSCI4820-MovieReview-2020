import os, sys
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from random import sample

def renameFiles(dir,newNamePrefix):
    number=1
    for file in os.listdir(dir):
        newName=newNamePrefix+str(number)
        os.rename(os.path.join(dir, file), os.path.join(dir, newName))
        number+=1

def gen_testData(dir,newNamePrefix,number):
    testFiles=sample(os.listdir(dir),number)
    for file in testFiles:
        newName="testing"+newNamePrefix+file
        os.rename(os.path.join(dir, file), os.path.join(dir, newName))

def make_Dictionary(train_dir):
    reviews = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for review in reviews:
        with open(review) as m:
            for i,line in enumerate(m):
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    return dictionary

def make_DictionaryNL(train_dir):
    stop_words  = set(stopwords.words('english'))
    neg_words=set(["haven't","won't","couldn't","isn't", "shouldn't",  "wouldn't","doesn't", "wasn't", "didn't", "cannot","can't","not"])
    review_stop_words=stop_words.difference(neg_words)
    print(review_stop_words)
    reviews = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for review in reviews:
        with open(review) as m:
            for i,line in enumerate(m):
                words = line.split()
                filteredWords = [w.lower() for w in words if not w.lower() in review_stop_words]
                all_words += filteredWords
    dictionary = Counter(all_words)
    return dictionary

def removeNoMeaningWords (wordDic):
    list_to_remove = wordDic.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del wordDic[item]
        elif (len(item) <=2 or item=="film" or item=="movie"):
            del wordDic[item]

    wordCommonDic = wordDic.most_common(3000)
    return wordCommonDic

def extract_features(dir,wordDic):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for file in files:
      with open(file) as fi:
        for i,line in enumerate(fi):
          words = line.split()
          for word in words: # 3 words
             wordID = 0
             for i,d in enumerate(wordDic):
                 if d[0] == word:
                     wordID = i
                    # features_matrix[docID,wordID] = words.count(word)
                     features_matrix[docID, wordID] +=1
        docID = docID + 1
    return features_matrix
