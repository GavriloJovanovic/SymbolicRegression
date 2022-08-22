import numpy as np
import SymbolicRegression
import pdb
import random


def f(x):
    return x * x * x * x - 4 * x * x * x + x * x - 5 * x + 1

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

    X = np.linspace(1, 10, 1000)
    # def __init__(self,goals,POPULATION_NUMBER,ITERATION_NUMBER,TOURNAMENT_SIZE,ELITISM_SIZE,MUTATION_RATE)
    # random.seed(12345)



    
    niz = [round((x - 30) / 10, 3) for x in range(0, 61, 2)]
    trening = [[niz[i],f(niz[i])] for i in range(0,31)]
    drvo = SymbolicRegression.GP(trening, 1000, 40, 5, 200, 0.2)
    drvo.printPopulation()
    print("POCINJEMO")
    print("============================================================")
    # pdb.set_trace()
    drvo.GP()
    print("POBEDNIK JE")
    print(drvo.population[0][0].stringNode() + " a njegov fitness je: " + str(drvo.population[0][1]))
    '''
    niz = [round((x - 30) / 10, 3) for x in range(0, 61, 2)]
    trening = [[niz[i],f(niz[i])] for i in range(0,31)]
    drvo = Node.GP(trening, 2, 1000, 10, 50, 0.2)
    drvo.onlyMutate()
    '''

except IOError:
    print("Greska")