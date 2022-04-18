import numpy as np
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
    print(drvo.tree2.stringNode()
    '''

    X = np.linspace(1,10,1000)
    #def __init__(self,goals,POPULATION_NUMBER,ITERATION_NUMBER,TOURNAMENT_SIZE,ELITISM_SIZE,MUTATION_RATE)
    trening = [[0.5, 0.9974389465414235], [1, 1.3817732906760363], [1.5, 2.3151009215268252], [2, 3.2210428707555843], [2.5, 2.939307285102794], [3, 0.2800875759383594], [3.5, -5.23355122648864], [4, -12.762483545790463], [4.5, -20.005780682148995], [5, -23.689444681115237], [5.5, -20.633925074213096], [6, -9.098787648510966], [6.5, 10.06540712243823], [7, 32.946245591563965], [7.5, 53.109134011414085], [8, 63.17342775008782], [8.5, 57.08868198436235], [9, 32.47046704269761], [9.5, -7.779560777874668]]

    #random.seed(12345)
    drvo = Node.GP(trening,20,100,4,2,1)
    drvo.printPopulation()
    print("============================================================")
    drvo.GP()


except IOError:
    print("Greska")