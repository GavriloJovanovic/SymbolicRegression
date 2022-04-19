"""
Module that has Node class for making Symbolic tree, and GP class that is
the algorithm of my project
"""
from enum import Enum
import random
import numpy as np
import sys


class Type(Enum):
    FIRST = 0
    TERM = 1
    OPERATOR = 2
    TRIGONOMETRY = 3


class Node:
    def __init__(self, type, level, char=None, value=None):
        self.type = type
        self.char = char
        self.value = value
        self.level = level
        self.child1 = None
        self.child2 = None

# RADI
    def getSubTreeFromPath(self, path, GP):
        if len(path) > 1:
            putSpot = path.pop(0)
            if putSpot == 'l' and self.child1 is not None:
                self.child1.getSubTreeFromPath(path, GP)
            elif putSpot == 'r' and self.child2 is not None:
                self.child2.getSubTreeFromPath(path, GP)
        else:
            putSpot = path.pop(0)
            if putSpot == 'l':
                GP.setCrossoverNode(self.child1)
            else:
                GP.setCrossoverNode(self.child2)

    def putSubTree(self, path, node):
        if len(path) > 1:
            putSpot = path.pop(0)
            if putSpot == 'l' and self.child1 is not None:
                self.child1.putSubTree(path, node)
            elif putSpot == 'r' and self.child2 is not None:
                self.child2.putSubTree(path, node)
        elif len(path) == 1:
            putSpot = path.pop(0)
            if putSpot == 'l':
                self.child1 = node
            else:
                self.child2 = node

# RADI
    def getPath(self, GP):
        if GP.numberForMakingLocalPath == 0:
            return
        else:
            if self.type == Type.FIRST:
                GP.numberForMakingLocalPath = GP.numberForMakingLocalPath - 1
                GP.localPath.append('l')
                self.child1.getPath(GP)
            elif self.type == Type.TRIGONOMETRY:
                GP.numberForMakingLocalPath = GP.numberForMakingLocalPath - 1
                GP.localPath.append('l')
                self.child1.getPath(GP)
                if GP.numberForMakingLocalPath > 0:
                    GP.localPath.pop()
            elif self.type == Type.OPERATOR:
                GP.numberForMakingLocalPath = GP.numberForMakingLocalPath - 1
                GP.localPath.append('l')
                self.child1.getPath(GP)
                if GP.numberForMakingLocalPath > 0:
                    GP.localPath.pop()
                if GP.numberForMakingLocalPath > 0:
                    GP.numberForMakingLocalPath = GP.numberForMakingLocalPath - 1
                    GP.localPath.append('r')
                    self.child2.getPath(GP)
                    if GP.numberForMakingLocalPath > 0:
                        GP.localPath.pop()
                else:
                    return

    def getRandomPath(self, GP):
        numberOfNodes = self.getDepthOfNode()
        arrayOfChoice = list(range(1, numberOfNodes + 1))
        GP.numberForMakingLocalPath = random.choice(arrayOfChoice)
        #print("Izabrao sam broj: " + str(GP.numberForMakingLocalPath))
        self.getPath(GP)

    # zapamtiti sve validne cvorove (set)
    # l pa d, broj covrova, [0,broj], 7

    def getSubTree(self, appendCrossoverProb, tree):
        if (self.type == Type.FIRST or self.type == Type.TRIGONOMETRY):
            randomNumber = random.random()
            tree.path.append('l')
            if randomNumber < tree.getCrossoverProbability():
                tree.setCrossoverProbability(0)
                tree.setFinalPath(tree.path)
                tree.setCrossoverNode(self.child1)
            else:
                tree.setCrossoverProbability(
                    tree.getCrossoverProbability() + appendCrossoverProb)
                self.child1.getSubTree(appendCrossoverProb, tree)
                if self.type == Type.TRIGONOMETRY and tree.getCrossoverProbability() > 0.001:
                    tree.path.pop()

        elif self.type == Type.OPERATOR:
            randomNumber = random.random()
            tree.path.append('l')
            if randomNumber < tree.getCrossoverProbability():
                tree.setCrossoverProbability(0)
                tree.setFinalPath(tree.path)
                tree.setCrossoverNode(self.child1)
            else:
                tree.setCrossoverProbability(
                    tree.getCrossoverProbability() + appendCrossoverProb)
                self.child1.getSubTree(appendCrossoverProb, tree)
                if tree.getCrossoverProbability() > 0.001:
                    tree.path.pop()

            if tree.getCrossoverProbability() > 0.001:
                randomNumber = random.random()
                tree.path.append('r')
                if randomNumber < tree.getCrossoverProbability():
                    tree.setCrossoverProbability(0)
                    tree.setFinalPath(tree.path)
                    tree.setCrossoverNode(self.child2)
                else:
                    tree.setCrossoverProbability(tree.getCrossoverProbability()
                                                 + appendCrossoverProb)
                    self.child2.getSubTree(appendCrossoverProb, tree)
                    if tree.getCrossoverProbability() > 0.001:
                        tree.path.pop()

    def setChild1(self, node):
        self.child1 = node

    def setChild2(self, node):
        self.child2 = node

    def getType(self):
        return self.type

    def stringNode(self):
        #print("TYPE U STRING NODE: "+ str(self.type))
        if self.type == Type.FIRST:
            return self.child1.stringNode()
        if self.type == Type.TERM:
            return self.char
        elif self.type == Type.TRIGONOMETRY:
            return self.char + "(" + self.child1.stringNode() + ")"
        else:
            return "( " + self.child1.stringNode() + " " \
                + self.char + " " + self.child2.stringNode() + " )"

    def mutateInPath(self, path, GP):
        if len(path) > 1:
            putSpot = path.pop(0)
            if putSpot == 'l' and self.child1 is not None:
                self.child1.getSubTreeFromPath(path, GP)
            elif putSpot == 'r' and self.child2 is not None:
                self.child2.getSubTreeFromPath(path, GP)
        else:
            putSpot = path.pop(0)
            if putSpot == 'l':
                GP.generateSubTree(None, 3, 3, 1)
                self.child1 = GP.getModTree()
            else:
                GP.generateSubTree(None, 3, 3, 1)
                self.child2 = GP.getModTree()

    def mutate(self, tree):
        # We are mutationg FIRST,TRIGONOMETRY and OPERATORS nodes becouse
        # we want to change our node, and subsequently our subtree

        # If our node is FIRST or TRIGONOMETRY we are only applying mutation on
        # first child
        if self.type == Type.FIRST or self.type == Type.TRIGONOMETRY:
            randomMutation = random.random()
            if randomMutation < tree.getMutationRate():
                tree.generateSubTree(None, 3, 3, 1)
                self.child1 = tree.getModTree()
                #print("Mutiram " + str(tree.getMutationRate()) + " : " + self.child1.stringNode())
                # We are allways setting setMutationRate on 0 after we
                # succesfully mutate a node
                tree.setMutationRate(0)
            else:
                tree.setMutationRate(tree.getMutationRate())
                # print(tree.getMutationRate())
                self.child1.mutate(tree)

        elif self.type == Type.OPERATOR:
            randomMutation = random.random()
            if randomMutation < tree.getMutationRate():
                tree.generateSubTree(None, 3, 3, 1)
                self.child1 = tree.getModTree()
                #print("Mutiram " + str(tree.getMutationRate()) + " : " + self.child1.stringNode())
                tree.setMutationRate(0)
            else:
                tree.setMutationRate(tree.getMutationRate())
                # print(tree.getMutationRate())
                self.child1.mutate(tree)

            randomMutation = random.random()
            if randomMutation < tree.getMutationRate():
                tree.generateSubTree(None, 3, 3, 1)
                self.child2 = tree.getModTree()
                #print("Mutiram " + str(tree.getMutationRate()) + " : " + self.child1.stringNode())
                tree.setMutationRate(0)
            else:
                tree.setMutationRate(tree.getMutationRate())
                # print(tree.getMutationRate())
                self.child2.mutate(tree)

    def getDepthOfNode(self):
        if self.type == Type.FIRST:
            return self.child1.getDepthOfNode()
        if self.type == Type.TERM:
            return 1
        if self.type == Type.TRIGONOMETRY:
            return 1 + self.child1.getDepthOfNode()
        if self.type == Type.OPERATOR:
            return 1 + self.child1.getDepthOfNode() + self.child2.getDepthOfNode()

    def getValue(self, xValue):
        if self.type == Type.FIRST:
            return self.child1.getValue(xValue)
        elif self.type == Type.TERM:
            if self.char == "x":
                return xValue
            return self.value
        elif self.type == Type.TRIGONOMETRY:
            if self.char == "cos":
                return np.cos(self.child1.getValue(xValue))
            else:
                return np.sin(self.child1.getValue(xValue))
        else:
            if self.char == "+":
                return self.child1.getValue(
                    xValue) + self.child2.getValue(xValue)
            elif self.char == "-":
                return self.child1.getValue(
                    xValue) - self.child2.getValue(xValue)
            elif self.char == "*":
                return self.child1.getValue(
                    xValue) * self.child2.getValue(xValue)
            else:
                return self.child1.getValue(
                    xValue) / (self.child2.getValue(xValue) + 0.000000001)


class GP:
    def __init__(self, goals, POPULATION_NUMBER, ITERATION_NUMBER,
                 TOURNAMENT_SIZE, ELITISM_SIZE, MUTATION_RATE):

        self.value = 0
        self.goals = goals  # that is array of vectors
        self.tree = None  # first is just one node, but it needs to be array of nodes
        self.tree2 = None
        self.ITERATION_NUMBER = ITERATION_NUMBER
        self.POPULATION_NUMBER = POPULATION_NUMBER
        self.ELITISM_SIZE = ELITISM_SIZE
        self.TOURNAMENT_SIZE = TOURNAMENT_SIZE
        self.population = []  # MATH IZRAZ + FITNESS
        # this is special modificiation node for making new subtrees
        self.modificationTree = None

        # attributes for crossover and mutation
        # We use this parametar to determan mutation rate of node
        self.MUTATION_RATE_INITIAL = MUTATION_RATE
        self.MUTATION_RATE = MUTATION_RATE
        self.CROSSOVER_PROBABILITY = 1
        self.path = []
        self.finalPath = []
        self.returnedNode = None

        self.localPath = []
        self.numberForMakingLocalPath = 0

        sys.setrecursionlimit(10000)

        self.createRandomPopulation()
        self.evaluateFirstFitness()
        #########

    def setPath(self, path):
        self.path = []
        for x in path:
            self.path.append(x)

    def evaluateFirstFitness(self):
        for i in range(self.POPULATION_NUMBER):
            self.population[i][1] = self.calculateFitness(i)

    def GP(self):
        # print("POCETAK")
        # for j in range(self.POPULATION_NUMBER):
        #print("Izraz broj. " + str(j) + " je " + self.population[j][0].stringNode() + " a fitness je: " + str(self.population[j][1]))
        # print("==========================================================================")

        newPopulation = []
        for i in range(self.ITERATION_NUMBER):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.population.sort(key=lambda tup: tup[1])
            #print("NAJBOLJE: Generacija " + str(i) + " izraz: " + self.population[0][0].stringNode() + " = " + str(self.calculateFitness(0)))
            #print("NAJGORE: Generacija " + str(i) + " izraz: " + self.population[-1][0].stringNode() + " = " + str(self.calculateFitness(self.POPULATION_NUMBER-1)))
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            #print("Pre promene u generaiciji")
            # for j in range(len(self.population)):
            #     print("Izraz broj. " + str(j) + " je " + self.population[j][0].stringNode() + " a fitness je: " + str(self.population[j][1]))

            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            j = 0
            while j < self.ELITISM_SIZE:
                newPopulation.append(self.population[j])
                #print("APENDUJEM U GENERACIJI: " + str(i) + " izraz: " + newPopulation[j][0].stringNode() + " -> " + str(self.calculateFitness(j)))
                j = j + 1
            print("Duzina nove populacije: " + str(len(self.population)))
            # Having the one copy of the poplations
            copyPopulation = []
            for x in self.population:
                copyPopulation.append(x)

            while j < self.POPULATION_NUMBER:
                copyPopulation = []
                for x in self.population:
                    copyPopulation.append(x)
                parent1Index = self.tournamentSelection()
                parent2Index = self.tournamentSelection()
                self.betterCrossover(parent1Index, parent2Index)
                self.betterMutation(parent1Index)
                self.betterMutation(parent2Index)
                self.population[parent1Index][1] = self.calculateFitness(
                    parent1Index)
                self.population[parent2Index][1] = self.calculateFitness(
                    parent2Index)
                newPopulation.append(self.population[parent1Index])
                newPopulation.append(self.population[parent2Index])
                #print("APENDUJEM U GENERACIJI: " + str(i) + " izraz: " + newPopulation[j][0].stringNode()+ " -> " + str(self.calculateFitness(j)))
                #print("APENDUJEM U GENERACIJI: " + str(i) + " izraz: " + newPopulation[j+1][0].stringNode()+ " -> " + str(self.calculateFitness(j+1)))
                #print("APENDUJEMO: ")
                # print(newPopulation[j][0].stringNode())
                # print(newPopulation[j+1][0].stringNode())

                # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                j = j + 2
                self.population = []
                for x in copyPopulation:
                    self.population.append(x)

            self.population = newPopulation
            newPopulation = []
            print("Duzina populacije nakon tranformacije: " +
                  str(len(self.population)))

            for j in range(len(self.population)):
                print("Izraz broj. " +
                      str(j) +
                      " je " +
                      self.population[j][0].stringNode() +
                      " a fitness je: " +
                      str(self.population[j][1]))
            #print("STARA POPULACIJA")
            # for j in range(self.POPULATION_NUMBER):
            #    print("Izraz broj. " + str(j) + " je " + self.population[j][0].stringNode() + " a fitness je: " + str(self.population[j][1]))
            # print("===================================================================================")
            #print("NOVA POPULACIJA")
            # for j in range(self.POPULATION_NUMBER):
            #    print("Izraz broj. " + str(j) + " je " + newPopulation[j][0].stringNode() + " a fitness je: " + str(newPopulation[j][1]))

            # print("======================================================================")
            #print("Ispisujemo izrze")
            # for j in range(self.POPULATION_NUMBER):
                #print("Izraz broj. " + str(j) + " je " + self.population[j][0].stringNode() + " a fitness je: " + str(self.population[j][1]))

            #print("PROVERAVAMO JEDNAKOST: ", end="")
            #bul = True
            # for j in range(self.POPULATION_NUMBER):
                # if self.population[j] != newPopulation[j]:
                #bul = False
            # print(bul)

    def tournamentSelection(self):
        self.TOURNAMENT_SIZE = 5
        listIndexes = list(range(self.POPULATION_NUMBER))
        participants = random.choices(listIndexes, k=self.TOURNAMENT_SIZE)
        minNumber = float("inf")
        returnIndex = 0

        for i in range(self.TOURNAMENT_SIZE):
            if self.population[participants[i]][1] < minNumber:
                returnIndex = participants[i]
        return returnIndex

    def selection(self):
        iterationIndex = self.SELECTION_NUMBER
        print("BEZ MUTACIJE SELEKCIJA")
        for i in range(self.POPULATION_NUMBER):
            print(self.population[i][0].stringNode())
            print("----------------------------------")
        print("------------------------------")
        while True:
            self.crossover(iterationIndex, iterationIndex + 1)
            iterationIndex += 2
            if iterationIndex == self.POPULATION_NUMBER:
                break
        print("POSLE MUTACIJA SELEKCIJA")
        for i in range(self.POPULATION_NUMBER):
            if i == self.SELECTION_NUMBER:
                print("MUTIRANI")
            print(self.population[i][0].stringNode())
            print("----------------------------------")
        print("------------------------------")

    def createRandomPopulation(self):
        for i in range(self.POPULATION_NUMBER):
            self.generateSubTree(None, 3, 3, 1, True)
            self.population.append([self.modificationTree, 0])

    def getPopulation(self):
        return self.population

    def getPopulationIndex(self, index):
        return self.population[index]

    def setCrossoverNode(self, node):
        self.returnedNode = node

    def getCrossoverNode(self):
        return self.returnedNode

    def setFinalPath(self, x):
        self.finalPath = x

    def betterMutation(self, index):
        randomNumber = random.random()
        if self.MUTATION_RATE > randomNumber:
            self.localPath = []
            self.population[index][0].getRandomPath(self)
            self.setPath(self.localPath)
            self.population[index][0].mutateInPath(self.localPath, self)

    def betterCrossover(self, index1, index2):
        # UZIMAMO NASUMICNO PODSTABLO1
        self.localPath = []
        self.setCrossoverNode(None)
        self.population[index1][0].getRandomPath(self)
        self.setPath(self.localPath)
        self.population[index1][0].getSubTreeFromPath(self.localPath, self)
        subtree1 = self.getCrossoverNode()
        path1 = self.path
        self.localPath = []
        self.setCrossoverNode(None)
        self.population[index2][0].getRandomPath(self)
        self.setPath(self.localPath)
        self.population[index2][0].getSubTreeFromPath(self.localPath, self)
        subtree2 = self.getCrossoverNode()
        path2 = self.path
        self.population[index2][0].putSubTree(path2, subtree1)
        self.population[index1][0].putSubTree(path1, subtree2)

    def crossover(self, index1, index2):
        self.path = []
        self.finalPath = []
        appendNode = self.population[index1][0].getDepthOfNode()
        self.setCrossoverProbabilityWithoutCheck(1 / appendNode)
        self.population[index1][0].getSubTree(1 / appendNode, self)
        path1 = self.finalPath
        subTree1 = self.getCrossoverNode()
        print("#1")
        print(str(path1))
        print(subTree1.stringNode())
        print("------------------------------------------------")
        ###
        self.path = []
        self.finalPath = []
        self.setCrossoverNode(None)
        appendNode = self.population[index2][0].getDepthOfNode()
        self.setCrossoverProbabilityWithoutCheck(1 / appendNode)
        self.population[index2][0].getSubTree(1 / appendNode, self)
        path2 = self.finalPath
        subTree2 = self.getCrossoverNode()
        print("#2")
        print(path2)
        print(subTree2.stringNode())
        print("------------------------------------------------")
        ###
        self.population[index2][0].putSubTree(path2, subTree1)
        self.population[index1][0].putSubTree(path1, subTree2)

    def calculateFitnessInd(self):
        return self.calculateFitness(0)

    def getCrossoverProbability(self):
        return self.CROSSOVER_PROBABILITY

    def setCrossoverProbabilityWithoutCheck(self, number):
        self.CROSSOVER_PROBABILITY = number

    def setCrossoverProbability(self, number):
        if number > 0.95:
            self.CROSSOVER_PROBABILITY = 1
        if number == 0 or self.getCrossoverProbability() == 0:
            self.CROSSOVER_PROBABILITY = 0
        else:
            self.CROSSOVER_PROBABILITY = number

    def printLocalPath(self):
        print("Printing local path: " + str(self.localPath))

    def resetMutationRate(self):
        self.MUTATION_RATE = self.MUTATION_RATE_INITIAL

    def getMutationRate(self):
        return self.MUTATION_RATE

    def setMutationRate(self, num):
        self.MUTATION_RATE = num

    # racuna fitness, jos nepotreban
    def calculateFitness(self, index):
        err = 0
        testLength = len(self.goals)
        for i in range(testLength):
            err = err + (self.getPopulationIndex(index)[0].getValue(self.goals[i][0]) - self.goals[i][1]) \
                * (self.getPopulationIndex(index)[0].getValue(self.goals[i][0]) - self.goals[i][1])
        return err

    def setModificationTreeOnNone(self):
        self.modificationTree = None

    # We use mutation on one of our's roots nodes!
    def mutation(self, mutationRate):
        # Setting mutation rate in node
        self.setMutationRate(mutationRate)
        # recusrsive method mutation
        self.tree.mutate(self)

    # funkcija koja generise nasumican cvor

    def generateRandomNode(self, depth, xValue):
        randType = None
        if depth <= 1:
            randType = Type.TERM
        elif depth == 2:
            random1 = random.random()
            if random1 < 0.666667:
                randType = Type.OPERATOR
            else:
                randType = Type.TRIGONOMETRY
        else:
            random2 = random.random()
            if random2 < 0.5714:
                randType = Type.OPERATOR
            elif random2 >= 0.5714 and random2 < 0.8571:
                randType = Type.TRIGONOMETRY
            else:
                randType = Type.TERM

        if randType == Type.OPERATOR:
            randomOperator = random.choice(['+', '-', '*', '/'])
            return Node(Type.OPERATOR, -1, randomOperator)
        elif randType == Type.TRIGONOMETRY:
            randomTrig = random.choice(['cos', 'sin'])
            return Node(Type.TRIGONOMETRY, -1, randomTrig)

        elif randType == Type.TERM:
            randNumberTerm = random.random()
            if randNumberTerm < 0.5:
                randomTermNumber = round(random.random() * 10, 2)
                return Node(Type.TERM, -
                            1, str(randomTermNumber), randomTermNumber)
            else:
                return Node(Type.TERM, -1, 'x', xValue)

    def printPopulation(self):
        for x in self.population:
            print(x[0].stringNode())

    def setTree(self, node):
        self.tree = node

    def setTree2(self, node):
        self.tree2 = node

    def getModTree(self):
        return self.modificationTree

    # funkcija koje generise nasumicno podstablo, pre nje moramo da kazemo
    # current node je trenutan cvor, depth je dubina cvora da znamo kako da konstruisemo nasumicni cvor,
    # nodeNum je slican broj, ignorisi ga a xValue je vrednost X-a koji
    # prosledjujemo za test prvo
    def generateSubTree(
            self,
            currentNode,
            depth,
            nodeNum,
            xValue,
            areWeGenerateFullTree=False):

        if currentNode is not None and currentNode.type == Type.TERM:
            return
        # Basic cases
        if nodeNum == 1 or depth == 1:
            return

        if currentNode is None:
            if areWeGenerateFullTree:
                self.modificationTree = None
                self.modificationTree = Node(Type.FIRST, 0)
                self.modificationTree.child1 = self.generateRandomNode(
                    depth, xValue)
                self.generateSubTree(
                    self.modificationTree.child1, depth, nodeNum, xValue)
            else:
                self.modificationTree = self.generateRandomNode(depth, xValue)
                self.generateSubTree(
                    self.modificationTree, depth, nodeNum, xValue)

        if currentNode is not None and currentNode.type == Type.TRIGONOMETRY:
            currentNode.child1 = self.generateRandomNode(depth - 1, xValue)
            self.generateSubTree(
                currentNode.child1,
                depth - 1,
                nodeNum - 1,
                xValue)

        if currentNode is not None and currentNode.type == Type.OPERATOR:
            depthFirst = depth - 2
            depthSecond = depth - 2
            depthForGenerating = depthFirst
            if depthFirst > nodeNum:
                depthForGenerating = nodeNum
            currentNode.child1 = self.generateRandomNode(
                depthForGenerating, xValue)
            self.generateSubTree(
                currentNode.child1,
                depthFirst,
                nodeNum - 1,
                xValue)

            depthForGenerating = depthSecond
            if depthSecond > nodeNum:
                depthForGenerating = nodeNum
            currentNode.child2 = self.generateRandomNode(
                depthForGenerating, xValue)
            self.generateSubTree(
                currentNode.child2,
                nodeNum - 1,
                nodeNum - 1,
                xValue)
