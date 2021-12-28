import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import Node



try:

    drvo = Node.Tree(0,5,0)
    drvo.generateSubTree(None,5,5,1,True)
    print("Prvi ispis, kako nam izgleda nasumicno drvo koje je napravljeno")
    print(drvo.modificationTree.stringNode())
    print("Prvi ispis, koja je vrednost drveta za x = 1")
    print(drvo.modificationTree.getValue(1))
    drvo.setTree(drvo.getModTree())
    print("Drugi ispis, postavli smo da je drvo modifikovano stablo iz generisanog sub tree")
    print(drvo.tree.stringNode())
    print("-------------------------------")
    drvo.mutation(0.3)
    print("Drvo posle mutacije")
    print(drvo.tree.stringNode())
    print(drvo.tree.getValue(1))
    print("---------------------------------------------")
    drvo.mutation(0.3)
    print("Drvo posle druge mutacije")
    print(drvo.tree.stringNode())
    print(drvo.tree.getValue(1))

except IOError:
    print("Greska")