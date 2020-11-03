import supporting as pr
import os, sys
import numpy as np
from collections import Counter

def main():
  # full_path = os.path.realpath(__file__)
  # dir=os.path.dirname(full_path)+"\\\dataStudents\\posStud"
  # pr.renameFiles(dir,"pos")
  #
  # dir = os.path.dirname(full_path) + "\\\dataStudents\\negStud"
  # pr.renameFiles(dir,"neg")
  #
  # print("Done!")

    full_path = os.path.realpath(__file__)
    dir=os.path.dirname(full_path)+"\\\dataStudents\\posStud"
    pr.gen_testData(dir,"",10)
    dir=os.path.dirname(full_path)+"\\\dataStudents\\negStud"
    pr.gen_testData(dir,"",10)
    print("done!")

if __name__ == '__main__':
    main()
