import os, sys
import numpy as np
from collections import Counter
import supporting as pr


def main():
  full_path = os.path.realpath(__file__)
  dir=os.path.dirname(full_path)+"\\\data\\training\\"
  #print(dir)
  #print("ok" + os.getcwd())
  dictionary=pr.make_DictionaryNL(dir)
  wordCommonDic = pr.removeNoMeaningWords(dictionary)

  print(wordCommonDic)
  features_matrix=pr.extract_features(dir,wordCommonDic)
  np.save("features.npy",features_matrix)
  print("Done!")

if __name__ == '__main__':
    main()
