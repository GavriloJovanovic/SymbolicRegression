from enum import Enum
import random
import numpy as np
import sys


file_path = 'output.txt'
sys.stdout = open(file_path, "w")

class Type(Enum):
    FIRST = 0
    TERM = 1
    OPERATOR = 2
    TRIGONOMETRY = 3


class Node:
    def __init__(self,type,level,char = None,value=None):
        self.type = type
        self.char = char
        self.value = value
        self.level = level
        self.child1 = None
        self.child2 = None

#RADI
    def getSubTreeFromPath(self,path,GP):
        if len(path) > 1:
            putSpot = path.pop(0)
            if putSpot == 'l' and self.child1 is not None:
                self.child1.getSubTreeFromPath(path,GP)
            elif putSpot == 'r' and self.child2 is not None:
                self.child2.getSubTreeFromPath(path,GP)
        else:
            putSpot = path.pop(0)
            if putSpot == 'l':
                GP.setCrossoverNode(self.child1)
            else:
                GP.setCrossoverNode(self.child2)

    def putSubTree(self,path,node):
        if len(path) > 1:
            putSpot = path.pop(0)
            if putSpot == 'l' and self.child1 is not None:
                self.child1.putSubTree(path,node)
            elif putSpot == 'r' and self.child2 is not None:
                self.child2.putSubTree(path,node)
        elif len(path) == 1:
            putSpot = path.pop(0)
            if putSpot == 'l':
                self.child1 = node
            else:
                self.child2 = node

#RADI
    def getPath(self,GP):
        if GP.numberForMakingLocalPath==0:
            return
        else:
            if self.type == Type.FIRST:
                GP.numberForMakingLocalPath = GP.numberForMakingLocalPath-1
                GP.localPath.append('l')
                self.child1.getPath(GP)
            elif self.type == Type.TRIGONOMETRY:
                GP.numberForMakingLocalPath = GP.numberForMakingLocalPath -1
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

    def getRandomPath(self,GP):
        numberOfNodes = self.getDepthOfNode()
        arrayOfChoice = list(range(1,numberOfNodes+1))
        GP.numberForMakingLocalPath = random.choice(arrayOfChoice)
        #print("Izabrao sam broj: " + str(GP.numberForMakingLocalPath))
        self.getPath(GP)


    def setChild1(self,node):
        self.child1 = node

    def setChild2(self,node):
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
            return "( " + self.child1.stringNode() + " " + self.char + " " + self.child2.stringNode() + " )"

    def mutateInPath(self,path,GP,nodeNumForMutation):
        if len(path) > 1:
            putSpot = path.pop(0)
            if putSpot == 'l' and self.child1 is not None:
                self.child1.mutateInPath(path,GP,nodeNumForMutation)
            elif putSpot == 'r' and self.child2 is not None:
                self.child2.mutateInPath(path,GP,nodeNumForMutation)
        else:
            putSpot = path.pop(0)
            if putSpot == 'l':
                if nodeNumForMutation <= 0:
                    nodeNumForMutation = 1
                randomDepth = random.randint(1,nodeNumForMutation)
                GP.depthOfNodes.setDepth(randomDepth)
                GP.generateSubTree(None,randomDepth,randomDepth,1)
                self.child1 = GP.getModTree()
            else:
                if nodeNumForMutation <= 0:
                    nodeNumForMutation = 1
                randomDepth = random.randint(1,nodeNumForMutation)
                GP.depthOfNodes.setDepth(randomDepth)
                GP.generateSubTree(None,randomDepth,randomDepth,1)
                self.child2 = GP.getModTree()


    def getDepthOfNode(self):
        if self.type == Type.FIRST:
            return self.child1.getDepthOfNode()
        if self.type == Type.TERM:
            return 1
        if self.type == Type.TRIGONOMETRY:
            return 1 + self.child1.getDepthOfNode()
        if self.type == Type.OPERATOR:
            return 1 + self.child1.getDepthOfNode() + self.child2.getDepthOfNode()

    def getValue(self,xValue):
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
                return self.child1.getValue(xValue) + self.child2.getValue(xValue)
            elif self.char == "-":
                return self.child1.getValue(xValue) - self.child2.getValue(xValue)
            elif self.char == "*":
                return self.child1.getValue(xValue) * self.child2.getValue(xValue)
            else:
                return self.child1.getValue(xValue) / (self.child2.getValue(xValue) + 0.000001)



 #   @abstractmethod
  #  def evaluete(self):


class GP:
    def __init__(self,goals,POPULATION_NUMBER,ITERATION_NUMBER,TOURNAMENT_SIZE,ELITISM_SIZE,MUTATION_RATE):
        self.value = 0
        self.goals = goals # that is array of vectors
        self.tree = None # first is just one node, but it needs to be array of nodes
        self.tree2 = None
        self.ITERATION_NUMBER = ITERATION_NUMBER
        self.POPULATION_NUMBER = POPULATION_NUMBER
        self.ELITISM_SIZE = ELITISM_SIZE
        self.TOURNAMENT_SIZE = TOURNAMENT_SIZE
        self.population = [] #MATH IZRAZ + FITNESS
        self.modificationTree = None #this is special modificiation node for making new subtrees
        self.MAX_NODE_NUMBER = 20

        # attributes for crossover and mutation
        self.MUTATION_RATE_INITIAL = MUTATION_RATE #We use this parametar to determan mutation rate of node
        self.MUTATION_RATE = MUTATION_RATE
        self.CROSSOVER_PROBABILITY = 1
        self.path = []
        self.finalPath = []
        self.returnedNode = None
        self.depthOfNodes = DepthOfNode(5)
        self.localPath = []
        self.numberForMakingLocalPath = 0

        sys.setrecursionlimit(6000)



        self.createRandomPopulation()
        self.evaluateFirstFitness()
        #########


    def setPath(self,path):
        self.path = []
        for x in path:
            self.path.append(x)

    def evaluateFirstFitness(self):
        for i in range(self.POPULATION_NUMBER):
            self.population[i][1] = self.calculateFitness(i)

    def GP(self):
        newPopulation = []

        for i in range(self.ITERATION_NUMBER):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.population.sort(key=lambda tup: tup[1])
            print("NAJBOLJE: Generacija " + str(i) + " izraz: " + self.population[0][0].stringNode() + " = " + str(self.calculateFitness(0)))
            print("NAJGORE: Generacija " + str(i) + " izraz: " + self.population[-1][0].stringNode() + " = " + str(self.calculateFitness(self.POPULATION_NUMBER-1)))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Pre promene u generaiciji: " + str(i))
            for j in range(len(self.population)):
                 print("Izraz broj. " + str(j) + " je " + self.population[j][0].stringNode() + " a fitness je: " + str(self.population[j][1]))

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            j = 0
            while j < self.ELITISM_SIZE:
                newPopulation.append(self.population[j])
                
                j = j + 1

            ### Having the one copy of the poplations
            copyPopulation = []
            for x in self.population:
                copyPopulation.append(x)

            while j < self.POPULATION_NUMBER:
                randomForCrossoverOrMutation = random.random()
                if randomForCrossoverOrMutation-2 > self.MUTATION_RATE and j != self.POPULATION_NUMBER-1:
                    parent1Index = j
                    parent2Index = j+1
                    self.betterCrossover(parent1Index,parent2Index)
                    self.population[parent1Index][1] = self.calculateFitness(parent1Index)
                    self.population[parent2Index][1] = self.calculateFitness(parent2Index)
                    newPopulation.append(self.population[parent1Index])
                    newPopulation.append(self.population[parent2Index])
                    j = j + 2
                    self.population = copyPopulation.copy()
                else:
                    #indexChosen = self.tournamentSelection()
                    self.betterMutation(j)
                    self.population[j][1] = self.calculateFitness(j)
                    newPopulation.append(self.population[j])
                    j = j + 1
                    self.population = copyPopulation.copy()

            self.population = newPopulation.copy()
            newPopulation = []

        print("Poslednja GLUPOST: ")
        for i in range(len(self.population)):
            print("Izraz broj. " + str(i) + " je " + self.population[i][0].stringNode() + " a fitness je: " + str(self.population[i][1]))


    def tournamentSelection(self):
        listIndexes = list(range(self.POPULATION_NUMBER))
        participants = random.choices(listIndexes,k=self.TOURNAMENT_SIZE)
        minNumber = float("inf")
        returnIndex = 0

        for i in range(self.TOURNAMENT_SIZE):
            if self.population[participants[i]][1] < minNumber:
                returnIndex = participants[i]
                minNumber = self.population[i][1]
        return returnIndex

    def createRandomPopulation(self):
        print("NASUMICNA POPULACIJA")
        for i in range(self.POPULATION_NUMBER):
            randomDepth = 5
            self.depthOfNodes.setDepth(randomDepth)
            self.generateSubTree(None, randomDepth, randomDepth, 1, True)
            self.population.append([self.modificationTree,0])
            #TO DO

    def getPopulation(self):
        return self.population

    def getPopulationIndex(self,index):
        return self.population[index]


    def setCrossoverNode(self,node):
        self.returnedNode = node

    def getCrossoverNode(self):
        return self.returnedNode

    def setFinalPath(self,x):
        self.finalPath = x

    def betterMutation(self,index):
        self.localPath = []
        numberOfNodesInIndex = self.population[index][0].getDepthOfNode()
        self.population[index][0].getRandomPath(self)
        self.setPath(self.localPath)
        #print(self.localPath)
        nodesToHaveInMutation = self.MAX_NODE_NUMBER - numberOfNodesInIndex - 1
        self.population[index][0].mutateInPath(self.localPath,self,nodesToHaveInMutation)


    def onlyMutate(self):
        for i in range(1000):
            newPopulation = []
            print("GENERACIJA " + str(i))
            for index in range(len(self.population)):
                self.betterMutation(index,self)
                self.population[index][1] = self.calculateFitness(index)
                print("IZMENJENO " + self.population[index][0].stringNode())
                newPopulation.append(self.population[index])
                self.setModificationTreeOnNone()
            self.population = newPopulation.copy()


    def betterCrossover(self,index1,index2):
        # UZIMAMO NASUMICNO PODSTABLO1

        #MORAMO DA VIDIMO KOLIKO CVOROVA UZIMAMO ZA CROSSOVER
        depthFirstTree = self.population[index1][0].getDepthOfNode()
        depthSecondTree = self.population[index2][0].getDepthOfNode()
        subtree1 = None
        subtree2 = None
        path1 = []
        path2 = []
        ########################################
        while True:
            self.localPath = []
            self.setCrossoverNode(None)
            self.population[index1][0].getRandomPath(self)
            self.setPath(self.localPath)
            self.population[index1][0].getSubTreeFromPath(self.localPath,self)
            subtree1 = self.getCrossoverNode()
            path1 = self.path
            if subtree1.getDepthOfNode() < self.MAX_NODE_NUMBER - depthSecondTree:
                break
        while True:
            self.localPath = []
            self.setCrossoverNode(None)
            self.population[index2][0].getRandomPath(self)
            self.setPath(self.localPath)
            self.population[index2][0].getSubTreeFromPath(self.localPath,self)
            subtree2 = self.getCrossoverNode()
            path2 = self.path
            if subtree2.getDepthOfNode() < self.MAX_NODE_NUMBER - depthFirstTree:
                break

        self.population[index2][0].putSubTree(path2,subtree1)
        self.population[index1][0].putSubTree(path1,subtree2)


    def calculateFitnessInd(self):
        return self.calculateFitness(0)


    def getCrossoverProbability(self):
        return self.CROSSOVER_PROBABILITY

    def setCrossoverProbabilityWithoutCheck(self,number):
        self.CROSSOVER_PROBABILITY = number

    def setCrossoverProbability(self,number):
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

    def setMutationRate(self,num):
        self.MUTATION_RATE = num

    #racuna fitness, jos nepotreban
    def calculateFitness(self,index):
        err = 0
        testLength = len(self.goals)
        for i in range(testLength):
            value = self.getPopulationIndex(index)[0].getValue(self.goals[i][0])
            err = err + np.abs(value- self.goals[i][1])
        #print()
        #print()
        #print("GOTOV JEDAN")
        #print("=======================================================================================")
        return err


    def setModificationTreeOnNone(self):
        self.modificationTree = None

    #We use mutation on one of our's roots nodes!
    def mutation(self,mutationRate):
        #Setting mutation rate in node
        self.setMutationRate(mutationRate)
        # recusrsive method mutation
        self.tree.mutate(self)


    # funkcija koja generise nasumican cvor
    def generateRandomNode(self, depth, xValue,isParentTrig = False):
        randType = None
        if depth <= 1:
            randType = Type.TERM
        elif depth == 2:
            random1 = random.random()
            if random1 < 0.333 and isParentTrig == True:
                randType = Type.TRIGONOMETRY
            else:
                randType = Type.TERM
        elif depth > 2:
            random2 = random.random()
            if random2 < 0.1 and isParentTrig == False:
                randType = Type.TRIGONOMETRY
            elif random2 >= 0.1 and random2 < 0.8:
                randType = Type.OPERATOR
            else:
                randType = Type.TERM

        if randType == Type.OPERATOR:
            randomOperator = random.choice(['+','+','+','+','-','-','*','*','*','/'])
            return Node(Type.OPERATOR,-1,randomOperator)
        elif randType == Type.TRIGONOMETRY:
            randomTrig = random.choice(['cos','cos','sin'])
            return Node(Type.TRIGONOMETRY,-1,randomTrig)


        elif randType == Type.TERM:
            randNumberTerm = random.random()
            if randNumberTerm < 0.8:
                randomTermNumber = round(random.random() * 20, 2)
                return Node(Type.TERM, -1, str(randomTermNumber), randomTermNumber)
            else:
                return Node(Type.TERM, -1, 'x', xValue)

    def valueForX(self,xValue):
        for x in self.population:
            value = x[0].getValue(xValue)
            print(x[0].stringNode() + " = " + str(value))

    def printPopulation(self):
        for x in self.population:
            print(x[0].stringNode() + ", fitness = " + str(x[1]))

    def setTree(self,node):
        self.tree = node

    def setTree2(self,node):
        self.tree2 = node

    def getModTree(self):
        return self.modificationTree


    #funkcija koje generise nasumicno podstablo, pre nje moramo da kazemo
    #current node je trenutan cvor, depth je dubina cvora da znamo kako da konstruisemo nasumicni cvor,
    #nodeNum je slican broj, ignorisi ga a xValue je vrednost X-a koji prosledjujemo za test prvo
    def generateSubTree(self,currentNode,depth,nodeNum,xValue,areWeGenerateFullTree = False):

        if currentNode != None and currentNode.type == Type.TERM:
            return
        ## Basic cases
        if self.depthOfNodes.getDepth() == 1:
            return


        if currentNode == None:
            if areWeGenerateFullTree == True:
                self.modificationTree = None
                self.modificationTree = Node(Type.FIRST,0)
                self.modificationTree.child1 = self.generateRandomNode(self.depthOfNodes.getDepth(),xValue)
                self.generateSubTree(self.modificationTree.child1,depth,nodeNum,xValue)
            else:
                self.modificationTree = None
                self.depthOfNodes.setDepth(self.depthOfNodes.getDepth()-1)
                self.modificationTree = self.generateRandomNode(self.depthOfNodes.getDepth(),xValue)
                self.generateSubTree(self.modificationTree,depth,nodeNum,xValue)



        if currentNode != None and currentNode.type == Type.TRIGONOMETRY:
            self.depthOfNodes.setDepth(self.depthOfNodes.getDepth() - 1)
            currentNode.child1 = self.generateRandomNode(self.depthOfNodes.getDepth(),xValue,True)
            self.generateSubTree(currentNode.child1,depth-1,nodeNum-1,xValue)


        if currentNode != None and currentNode.type == Type.OPERATOR:
            depthFirst= depth-2
            depthSecond = depth-2
            depthForGenerating = depthFirst
            if depthFirst > nodeNum:
                depthForGenerating = nodeNum
            self.depthOfNodes.setDepth(self.depthOfNodes.getDepth() - 2)
            currentNode.child1 = self.generateRandomNode(self.depthOfNodes.getDepth(),xValue)
            self.generateSubTree(currentNode.child1,depthFirst,nodeNum-1,xValue)
            #depthForGenerating = depthOfNodes.getDepth()
            #if depthSecond > nodeNum:
            #    depthForGenerating = nodeNum
            currentNode.child2 = self.generateRandomNode(self.depthOfNodes.getDepth(),xValue)
            self.generateSubTree(currentNode.child2,nodeNum-1,nodeNum-1,xValue)

class DepthOfNode():
    def __init__(self,initDepth):
        self.initialDepth = initDepth
        self.depth = initDepth
    def getDepth(self):
        return self.depth
    def setDepth(self,d):
        self.depth = d

