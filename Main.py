import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import Node

import random


try:

    '''
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
    

    drvo = Node.Tree(0,5,0)
    drvo.generateSubTree(None, 5, 5, 1, True)
    drvo.setTree(drvo.getModTree())
    drvo.generateSubTree(None,5,5,1,True)
    drvo.setTree2(drvo.getModTree())
    print(drvo.tree.stringNode())
    print(drvo.tree2.stringNode())
    drvo.crossover()
    print(drvo.tree.stringNode())
    print(drvo.tree2.stringNode())
    '''

    X = np.linspace(1,10,1000)
    #def __init__(self,goals,POPULATION_NUMBER,ITERATION_NUMBER,TOURNAMENT_SIZE,ELITISM_SIZE,MUTATION_RATE)
    trening = [[1,-10],[2,-8],[3,-4],[4,4],[5,15],[6,25],[7,49],[8,67]]
    #random.seed(12345)
    drvo = Node.GP(trening,2,1000,2,2,0.05)
    drvo.printPopulation()
    print("============================================================")
    drvo.betterCrossover(0,1)
    drvo.printPopulation()
    #drvo.GP()


except IOError:
    print("Greska")