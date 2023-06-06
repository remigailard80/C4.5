#!/usr/bin/env python
import pdb
from c45 import C45

file_name = input("insert file name: ")

c1 = C45("/content/C4.5/data/{0}/{1}.data".format(file_name, file_name), "/content/C4.5/data/{0}/{1}.names".format(file_name, file_name))
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()
