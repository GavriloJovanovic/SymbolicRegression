import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import Node



try:

    iteration = 0
    drvo = Node.Tree(0,5,0)
    drvo.generateSubTree(None,5,5,1,iteration)
    print(drvo.modificationTree.stringNode())
    print(drvo.modificationTree.getValue(1))

except IOError:
    print("Greska")