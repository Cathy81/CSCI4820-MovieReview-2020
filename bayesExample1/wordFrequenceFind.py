import os, sys
import numpy as np
from collections import Counter
import supporting as pr


def main():
  full_path = os.path.realpath(__file__)
  dir=os.path.dirname(full_path)+"\\\dataStudents\\learn\\neg"
  #print(dir)
  #print("ok" + os.getcwd())
  dictionary=pr.make_DictionaryNL(dir)
  wordCommonDic = pr.removeNoMeaningWords(dictionary)

  print(wordCommonDic)
  word = input("Enter the word.")
  while(word!=""):
    dic=dict(wordCommonDic)
    if(word in dic.keys()):
       print(dic[word])
    else:
       print("Not found")
    word = input("Enter the word.")


if __name__ == '__main__':
    main()
