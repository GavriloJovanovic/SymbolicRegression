from enum import Enum
import random
import numpy as np
from abc import ABC, abstractmethod

class Type(Enum):
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



    def setChild1(self,node):
        self.child1 = node

    def setChild2(self,node):
        self.child2 = node

    def getType(self):
        return self.type

    def stringNode(self):
        if self.type == Type.TERM:
            return self.char
        elif self.type == Type.TRIGONOMETRY:
            return self.char + "(" + self.child1.stringNode() + ")"
        else:
            return "( " + self.child1.stringNode() + " " + self.char + " " + self.child2.stringNode() + " )"

    def getValue(self,xValue):
        if self.type == Type.TERM:
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
                    exit(-1,"Greska")


 #   @abstractmethod
  #  def evaluete(self):


class Tree:
    def __init__(self,value,maxDepth,goals):
        self.value = 0
        self.maxDepth = maxDepth
        self.goals = goals # that is array of vectors
        self.tree = None
        self.modificationTree = None

    def calculateFitness(self):
        err = 0
        testLength = len(self.goals)
        for i in range(testLength):
            err = err + (self.nodes[0].getValue(self.goals[i][0]) - self.goals[i][1]) \
                  * (self.nodes[0].getValue(self.goals[i][0]) - self.goals[i][1])
        return err


    def setModificationTreeOnNone(self):
        self.modificationTree = None

    def generateRandomNode(self, depth, xValue):
        randType = None
        #print("I am generating a random number for depth: " + str(depth) )
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
        else:
            randNumberTerm = random.random()
            if randNumberTerm < 0.5:
                randomTermNumber = round(random.random() * 10, 2)
                return Node(Type.TERM, -1, str(randomTermNumber), randomTermNumber)
            else:
                return Node(Type.TERM, -1, 'x', xValue)

    # function that generate random subtree with <=depth elements
    # Before we call this function we must set modificationTree on None!
    def generateSubTree(self,currentNode,depth,nodeNum,xValue,iteration):
        #if currentNode != None:
            #print(currentNode.char + " " + str(iteration))

        if currentNode != None and currentNode.type == Type.TERM:
            return
        ## Basic cases
        if nodeNum  ==  1 or depth == 1:
            return


        if self.modificationTree == None:
            self.modificationTree = self.generateRandomNode(depth,xValue)
            self.generateSubTree(self.modificationTree,depth,nodeNum,xValue,iteration)



        if currentNode != None and currentNode.type == Type.TRIGONOMETRY:
            currentNode.child1 = self.generateRandomNode(depth-1,xValue)
            self.generateSubTree(currentNode.child1,depth-1,nodeNum-1,xValue,iteration+1)


        if currentNode != None and currentNode.type == Type.OPERATOR:
            depthFirst= depth-2
            depthSecond = depth-2
            depthForGenerating = depthFirst
            if depthFirst > nodeNum:
                depthForGenerating = nodeNum
            # print(str(depthForGenerating) + " iter:" + str(iteration))
            currentNode.child1 = self.generateRandomNode(depthForGenerating,xValue)
            self.generateSubTree(currentNode.child1,depthFirst,nodeNum-1,xValue,iteration+1)

            depthForGenerating = depthSecond
            if depthSecond > nodeNum:
                depthForGenerating = nodeNum
            # print(str(depthForGenerating) + " iter:" + str(iteration))
            currentNode.child2 = self.generateRandomNode(depthForGenerating,xValue)
            self.generateSubTree(currentNode.child2,nodeNum-1,nodeNum-1,xValue,iteration+1)





