from enum import Enum
import random
import numpy as np

class Type(Enum):
    FIRST = 0
    TERM = 1
    OPERATOR = 2
    TRIGONOMETRY = 3

SHIT_HAPPENDED = False


class Node:
    def __init__(self,type,level,char = None,value=None):
        self.type = type
        self.char = char
        self.value = value
        self.level = level
        self.child1 = None
        self.child2 = None
        self.MUTATION_RATE = 0 #We use this parametar to determan mutation rate of node
        self.CROSSOVER_PROBABILITY = 1


    def getCrossoverProbability(self):
        return self.CROSSOVER_PROBABILITY

    def setCrossoverProbability(self,number,isOver = False):
        if number > 0.95:
            self.CROSSOVER_PROBABILITY = 1
        if SHIT_HAPPENDED == True:
            self.CROSSOVER_PROBABILITY = 0
        else:
            self.CROSSOVER_PROBABILITY = number

    def getSubTree(self,path,appendCrossoverProb):
        elem = False
        if self.type == Type.FIRST or self.type == Type.TRIGONOMETRY:
            randomNumber = random.random()
            path.append('l')
            print(str(randomNumber) + " : " + str(self.getCrossoverProbability()) + " u Node " + str(self.type) + " " + str(path))
            if randomNumber < self.getCrossoverProbability():
                print("THAT HAPPENED")
                self.setCrossoverProbability(0)
                print(self.child1 == None)
                print(path)
                SHIT_HAPPENDED = True
                return path
            else:
                self.child1.setCrossoverProbability(self.getCrossoverProbability() + appendCrossoverProb)
                self.child1.getSubTree(path,appendCrossoverProb)
                if self.type == Type.TRIGONOMETRY:
                    print(str(self.type))
                    print("POPUJEM " + path.pop() +  " " + str(self.type))
        elif self.type == Type.OPERATOR:
            randomNumber = random.random()
            path.append('l')
            print(str(randomNumber) + " : " + str(self.getCrossoverProbability()) + " u Node " + str(self.type) + " " + str(path))
            if randomNumber < self.getCrossoverProbability():
                print("THAT HAPPENED")
                self.setCrossoverProbability(0)
                print(self.child1 == None)
                print(path)
                SHIT_HAPPENDED = True
                elem = True
                return path
            else:
                self.child1.setCrossoverProbability(self.getCrossoverProbability() + appendCrossoverProb)
                self.child1.getSubTree(path,appendCrossoverProb)
                print(str(self.type))
            if elem == False:
                randomNumber = random.random()
                path.append('r')
                print(str(randomNumber) + " : " + str(self.getCrossoverProbability()) + " u Node " + str(self.type) + " " + str(path))
                if randomNumber < self.getCrossoverProbability():
                    print("THAT HAPPENED")
                    self.setCrossoverProbability(0)
                    print(self.child2 == None)
                    print(path)
                    SHIT_HAPPENDED = True
                    return path
                else:
                    self.child2.setCrossoverProbability(self.getCrossoverProbability() + appendCrossoverProb)
                    self.child2.getSubTree(path,appendCrossoverProb)
                    print(str(self.type))
                    print("POPUJEM " + path.pop())
        else:
            print(str(self.type))
            print("POPUJEM " + path.pop())


    def getMutationRate(self):
        return self.MUTATION_RATE

    def setMutationRate(self,num):
        self.MUTATION_RATE = num

    def setChild1(self,node):
        self.child1 = node

    def setChild2(self,node):
        self.child2 = node

    def getType(self):
        return self.type

    def stringNode(self):
        if self.type == Type.FIRST:
            return self.child1.stringNode()
        if self.type == Type.TERM:
            return self.char
        elif self.type == Type.TRIGONOMETRY:
            return self.char + "(" + self.child1.stringNode() + ")"
        else:
            return "( " + self.child1.stringNode() + " " + self.char + " " + self.child2.stringNode() + " )"

    def mutate(self,Tree):
        # We are mutationg FIRST,TRIGONOMETRY and OPERATORS nodes becouse
        # we want to change our node, and subsequently our subtree

        #If our node is FIRST or TRIGONOMETRY we are only applying mutation on
        #first child
        if self.type == Type.FIRST or self.type == Type.TRIGONOMETRY:
            randomMutation = random.random()
            if randomMutation < self.getMutationRate():
                Tree.generateSubTree(None,3,3,1)
                self.child1 = Tree.getModTree()
                #We are allways setting setMutationRate on 0 after we succesfully mutate a node
                self.setMutationRate(0)
            else:
                self.child1.setMutationRate(self.getMutationRate())
                self.child1.mutate(Tree)

        elif self.type == Type.OPERATOR:
            randomMutation = random.random()
            if randomMutation < self.getMutationRate():
                Tree.generateSubTree(None,3,3,1)
                self.child1 = Tree.getModTree()
                self.setMutationRate(0)
            else:
                self.child1.setMutationRate(self.getMutationRate())
                self.child1.mutate(Tree)

            randomMutation = random.random()
            if randomMutation < self.getMutationRate():
                Tree.generateSubTree(None,3,3,1)
                self.child2 = Tree.getModTree()
                self.setMutationRate(0)
            else:
                self.child2.setMutationRate(self.getMutationRate())
                self.child2.mutate(Tree)

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
                if self.child2.getValue(xValue) != 0:
                    return self.child1.getValue(xValue) / self.child2.getValue(xValue)
                else:
                    exit(-1)



 #   @abstractmethod
  #  def evaluete(self):


class Tree:
    def __init__(self,value,maxDepth,goals):
        self.value = 0
        self.maxDepth = maxDepth
        self.goals = goals # that is array of vectors
        self.tree = None # first is just one node, but it needs to be array of nodes
        self.tree2 = None
        self.modificationTree = None #this is special modificiation node for making new subtrees


    #racuna fitness, jos nepotreban
    def calculateFitness(self):
        err = 0
        testLength = len(self.goals)
        for i in range(testLength):
            err = err + (self.nodes[0].getValue(self.goals[i][0]) - self.goals[i][1]) \
                  * (self.nodes[0].getValue(self.goals[i][0]) - self.goals[i][1])
        return err


    def setModificationTreeOnNone(self):
        self.modificationTree = None


    def crossover(self):
        depth1 = self.tree.getDepthOfNode()
        depth2 = self.tree2.getDepthOfNode()
        self.tree.setCrossoverProbability(1/(depth1+1))
        self.tree2.setCrossoverProbability(1 / (depth2 + 1))
        firstSubTree,path1 = self.tree.getSubTree([],depth1+1)
        secondSubTree,path2 = self.tree.getSubTree([],depth2+1)
        self.tree.putCrossoverSubTree(path1,secondSubTree)
        self.tree2.putCrossoverSubTree(path2,firstSubTree)

    #We use mutation on one of our's roots nodes!
    def mutation(self,mutationRate):
        #Setting mutation rate in node
        self.tree.setMutationRate(mutationRate)
        # recusrsive method mutation
        self.tree.mutate(self)


    # funkcija koja generise nasumican cvor
    def generateRandomNode(self, depth, xValue):
        randType = None
        if depth == 1:
            randType = Type.TERM
        elif depth == 2:
            randType = random.choice([Type.TERM, Type.TRIGONOMETRY])
        else:
            randType = random.choice([Type.TERM,Type.TRIGONOMETRY,Type.OPERATOR])

        if randType == Type.OPERATOR:
            randomOperator = random.choice(['+','-','*','/'])
            return Node(Type.OPERATOR,-1,randomOperator)
        elif randType == Type.TRIGONOMETRY:
            randomTrig = random.choice(['cos','sin'])
            return Node(Type.TRIGONOMETRY,-1,randomTrig)
        elif randType == Type.TERM:
            randNumberTerm = random.random()
            if randNumberTerm < 0.5:
                randomTermNumber = round(random.random() * 10, 2)
                return Node(Type.TERM, -1, str(randomTermNumber), randomTermNumber)
            else:
                return Node(Type.TERM, -1, 'x', xValue)

    def setTree(self,node):
        self.tree = node

    def getModTree(self):
        return self.modificationTree


    #funkcija koje generise nasumicno podstablo, pre nje moramo da kazemo
    #current node je trenutan cvor, depth je dubina cvora da znamo kako da konstruisemo nasumicni cvor,
    #nodeNum je slican broj, ignorisi ga a xValue je vrednost X-a koji prosledjujemo za test prvo
    def generateSubTree(self,currentNode,depth,nodeNum,xValue,areWeGenerateFullTree = False):

        if currentNode != None and currentNode.type == Type.TERM:
            return
        ## Basic cases
        if nodeNum  ==  1 or depth == 1:
            return


        if currentNode == None:
            if areWeGenerateFullTree == True:
                self.modificationTree = Node(Type.FIRST,0)
                self.modificationTree.child1 = self.generateRandomNode(depth,xValue)
                self.generateSubTree(self.modificationTree.child1,depth,nodeNum,xValue)
            else:
                self.modificationTree = self.generateRandomNode(depth,xValue)
                self.generateSubTree(self.modificationTree,depth,nodeNum,xValue)



        if currentNode != None and currentNode.type == Type.TRIGONOMETRY:
            currentNode.child1 = self.generateRandomNode(depth-1,xValue)
            self.generateSubTree(currentNode.child1,depth-1,nodeNum-1,xValue)


        if currentNode != None and currentNode.type == Type.OPERATOR:
            depthFirst= depth-2
            depthSecond = depth-2
            depthForGenerating = depthFirst
            if depthFirst > nodeNum:
                depthForGenerating = nodeNum
            currentNode.child1 = self.generateRandomNode(depthForGenerating,xValue)
            self.generateSubTree(currentNode.child1,depthFirst,nodeNum-1,xValue)

            depthForGenerating = depthSecond
            if depthSecond > nodeNum:
                depthForGenerating = nodeNum
            currentNode.child2 = self.generateRandomNode(depthForGenerating,xValue)
            self.generateSubTree(currentNode.child2,nodeNum-1,nodeNum-1,xValue)