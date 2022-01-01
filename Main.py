import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import Node



try:

    drvo = Node.Tree(0,5,0)
    drvo.generateSubTree(None,12,12,1,True)
    print("Prvi ispis, kako nam izgleda nasumicno drvo koje je napravljeno")
    print(drvo.modificationTree.stringNode())
    print("Prvi ispis, koja je vrednost drveta za x = 1")
    print(drvo.modificationTree.getValue(1))
    drvo.setTree(drvo.getModTree())
    print("Drugi ispis, postavli smo da je drvo modifikovano stablo iz generisanog sub tree")
    print(drvo.tree.stringNode())
    print("===================================================")
    appendNod = drvo.tree.getDepthOfNode()
    path = []
    drvo.setCrossoverProbability(1/appendNod)
    path,x = drvo.getSubTree(1/appendNod)
    print("Putanja je" + str(path))
    print(x.stringNode())


except IOError:
    print("Greska")