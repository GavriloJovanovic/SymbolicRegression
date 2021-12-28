from enum import Enum
import random
import numpy as np

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
        self.MUTATION_RATE = 0 #We use this parametar to determan mutation rate of node

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