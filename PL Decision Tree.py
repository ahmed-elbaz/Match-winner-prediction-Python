# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:16:07 2019

@author: Baz-PC
"""

###############
#Used libraries
###############

import os
import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#######################
# reading training data
#######################

os.chdir("E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 3")

# read all columns except for the first two which we will not use
usedCols=list(range(2,15))
training=pd.read_excel('Training_Data.xlsx',usecols=usedCols)

##########################
#Defining Entropy Function
##########################
def Entropy(pPlus,pMinus):
    if pPlus==0 or pMinus==0:
        return 0
    return -(pPlus)*math.log2(pPlus)-(pMinus)*math.log2(pMinus)

###############################################
#Discretizing columns for root node calcualtion
###############################################

# take a copy of the original dataFrame and discretize columns values for root node calcualtion
trainingRoot = training.copy()

#Discretinzing all columns except for the last column that shows match result
for column in range(trainingRoot.shape[1]-1):
    #print(column)
    colMin=np.min(trainingRoot.iloc[:,column])
    colMean=np.mean(trainingRoot.iloc[:,column])
    colMax=np.max(trainingRoot.iloc[:,column])
    trainingRoot.iloc[:,column]=pd.cut(trainingRoot.iloc[:,column],bins=[colMin,colMean,colMax],labels=["low","high"],include_lowest=True)

#################################################################
#calcualating the gain for all features to identify the root node
#################################################################  
# caculate entropy for FTR noting that cloumn 12 represents FTR
pHome=len(trainingRoot[trainingRoot.iloc[:,12]=='H'])/len(trainingRoot.iloc[:,12])
pAway=len(trainingRoot[trainingRoot.iloc[:,12]=='A'])/len(trainingRoot.iloc[:,12])
entropyS= Entropy(pHome,pAway)

# calculate gains for each attribute noting that cloumn 12 represents FTR
totalGain=[]
for column in range(trainingRoot.shape[1]-1):
    
    propHigh=len(trainingRoot[trainingRoot.iloc[:,column]=='high'])/len(trainingRoot.iloc[:,column])
    propLow=len(trainingRoot[trainingRoot.iloc[:,column]=='low'])/len(trainingRoot.iloc[:,column])
    
    pHighHome=len(trainingRoot[(trainingRoot.iloc[:,column]=='high') & (trainingRoot.iloc[:,12]=='H')])/len(trainingRoot[trainingRoot.iloc[:,column]=='high'])
    pHighAway=len(trainingRoot[(trainingRoot.iloc[:,column]=='high') & (trainingRoot.iloc[:,12]=='A')])/len(trainingRoot[trainingRoot.iloc[:,column]=='high'])
    
    pLowHome=len(trainingRoot[(trainingRoot.iloc[:,column]=='low') & (trainingRoot.iloc[:,12]=='H')])/len(trainingRoot[trainingRoot.iloc[:,column]=='low'])
    pLowAway=len(trainingRoot[(trainingRoot.iloc[:,column]=='low') & (trainingRoot.iloc[:,12]=='A')])/len(training[trainingRoot.iloc[:,column]=='low'])
    
    Gain=entropyS-propHigh*Entropy(pHighHome,pHighAway)-propLow*Entropy(pLowHome,pLowAway)
    totalGain.append(Gain)
  
print('The attribute with the highest gain for root node is '+trainingRoot.columns.values[np.argmax(totalGain)]+' and its value is '+str(np.max(totalGain)))
print('Number of times home win in case of high HST= '+str(len(trainingRoot[(trainingRoot.iloc[:,np.argmax(totalGain)]=='high') & (trainingRoot.iloc[:,12]=='H')])))
print('Number of times away win in case of high HST= '+str(len(trainingRoot[(trainingRoot.iloc[:,np.argmax(totalGain)]=='high') & (trainingRoot.iloc[:,12]=='A')])))
print('Number of times home win in case of low HST= '+str(len(trainingRoot[(trainingRoot.iloc[:,np.argmax(totalGain)]=='low') & (trainingRoot.iloc[:,12]=='H')])))
print('Number of times away win in case of low HST= '+str(len(trainingRoot[(trainingRoot.iloc[:,np.argmax(totalGain)]=='low') & (trainingRoot.iloc[:,12]=='A')])))

##################################################
#Discretizing columns for Second level calcualtion
##################################################

# take a copy of the original dataFrame and discretize columns values for second level calcualtion
trainingSecond = training.copy()

#Discretinzing all columns except for the last column that shows match result
for column in range(trainingSecond.shape[1]-1):
    colMin=np.min(trainingSecond.iloc[:,column])
    colMax=np.max(trainingSecond.iloc[:,column])
    Range=np.max(trainingSecond.iloc[:,column])-np.min(trainingSecond.iloc[:,column])
    colFirst=(Range/3)+colMin
    colSecond=Range*(2/3)+colMin
    trainingSecond.iloc[:,column]=pd.cut(trainingSecond.iloc[:,column],bins=[colMin,colFirst,colSecond,colMax],labels=["low","medium","high"],include_lowest=True)

#The above will discretize all columns to 3 levels but we need to keep HST as 2 levels only using the below command
trainingSecond.iloc[:,2]=trainingRoot.iloc[:,2]

#In the below part we will just change the order of columns in the dataframe
new_index=['HS', 'AS', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR','AR','HST','FTR']
trainingSecond=trainingSecond.reindex(new_index,axis=1)

####################################################################################
#calcualating the gain for all features to identify the second level splitting nodes
####################################################################################

###################################################################################################
#calculating the gains for low HST branch noting that cloumn 12 represents FTR and column 11 is HST
###################################################################################################
HSTLowHome=len(trainingSecond[(trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[trainingSecond.iloc[:,11]=='low'])
HSTLowAway=len(trainingSecond[(trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[trainingSecond.iloc[:,11]=='low'])
EntropyHST=Entropy(HSTLowHome,HSTLowAway)

LowHSTGain=[]
for column in range(trainingRoot.shape[1]-2):
    proplowHSTHigh=len(trainingSecond[(trainingSecond.iloc[:,column]=='high') & (trainingSecond.iloc[:,11]=='low') ])/len(trainingSecond[trainingSecond.iloc[:,11]=='low'])
    proplowHSTMedium=len(trainingSecond[(trainingSecond.iloc[:,column]=='medium') & (trainingSecond.iloc[:,11]=='low') ])/len(trainingSecond[trainingSecond.iloc[:,11]=='low'])
    proplowHSTLow=len(trainingSecond[(trainingSecond.iloc[:,column]=='low') & (trainingSecond.iloc[:,11]=='low') ])/len(trainingSecond[trainingSecond.iloc[:,11]=='low'])
    
    if len(trainingSecond[(trainingSecond.iloc[:,column]=='high')&(trainingSecond.iloc[:,11]=='low')])==0:
        plowHSTHighHome=0
        plowHSTHighAway=0
    else:
        plowHSTHighHome= len(trainingSecond[(trainingSecond.iloc[:,column]=='high') & (trainingSecond.iloc[:,11]=='low')& (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='high')&(trainingSecond.iloc[:,11]=='low')])
        plowHSTHighAway= len(trainingSecond[(trainingSecond.iloc[:,column]=='high') & (trainingSecond.iloc[:,11]=='low')& (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='high')&(trainingSecond.iloc[:,11]=='low')])
        
    if len(trainingSecond[(trainingSecond.iloc[:,column]=='medium')&(trainingSecond.iloc[:,11]=='low')])==0:
        plowHSTMediumHome=0
        plowHSTMediumAway=0
    else:
        plowHSTMediumHome= len(trainingSecond[(trainingSecond.iloc[:,column]=='medium') & (trainingSecond.iloc[:,11]=='low')& (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='medium')&(trainingSecond.iloc[:,11]=='low')])
        plowHSTMediumAway= len(trainingSecond[(trainingSecond.iloc[:,column]=='medium') & (trainingSecond.iloc[:,11]=='low')& (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='medium')&(trainingSecond.iloc[:,11]=='low')])
    
    if len(trainingSecond[(trainingSecond.iloc[:,column]=='low')&(trainingSecond.iloc[:,11]=='low')])==0:
        plowHSTLowHome=0
        plowHSTLowAway=0
    else: 
        plowHSTLowHome= len(trainingSecond[(trainingSecond.iloc[:,column]=='low') & (trainingSecond.iloc[:,11]=='low')& (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='low')&(trainingSecond.iloc[:,11]=='low')])
        plowHSTLowAway= len(trainingSecond[(trainingSecond.iloc[:,column]=='low') & (trainingSecond.iloc[:,11]=='low')& (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='low')&(trainingSecond.iloc[:,11]=='low')])
        
    GainSecond=EntropyHST-proplowHSTHigh*Entropy(plowHSTHighHome,plowHSTHighAway)-proplowHSTMedium*Entropy(plowHSTMediumHome,plowHSTMediumAway)-proplowHSTLow*Entropy(plowHSTLowHome,plowHSTLowAway)
    LowHSTGain.append(GainSecond)

print('The attribute with the highest gain for second Node with HST(the root node)is low is '+trainingSecond.columns.values[np.argmax(LowHSTGain)]+' and its value is '+str(np.max(LowHSTGain)))
print('Number of times home win in case of low HST and low LST = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(LowHSTGain)]=='low') & (trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='H')])))
print('Number of times away win in case of low HST and low LST = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(LowHSTGain)]=='low') & (trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='A')])))
print('Number of times home win in case of low HST and medium LST = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(LowHSTGain)]=='medium') & (trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='H')])))
print('Number of times away win in case of low HST and medium LST = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(LowHSTGain)]=='medium') & (trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='A')])))
print('Number of times home win in case of low HST and high LST = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(LowHSTGain)]=='high') & (trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='H')])))
print('Number of times away win in case of low HST and high LST = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(LowHSTGain)]=='high') & (trainingSecond.iloc[:,11]=='low') & (trainingSecond.iloc[:,12]=='A')])))

####################################################################################################
#calculating the gains for high HST branch noting that cloumn 12 represents FTR and column 11 is HST
####################################################################################################
HSTHighHome=len(trainingSecond[(trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[trainingSecond.iloc[:,11]=='high'])
HSTHighAway=len(trainingSecond[(trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[trainingSecond.iloc[:,11]=='high'])
EntropyHST=Entropy(HSTHighHome,HSTHighAway)

HighHSTGain=[]
for column in range(trainingRoot.shape[1]-2):
    propHighHSTHigh=len(trainingSecond[(trainingSecond.iloc[:,column]=='high') & (trainingSecond.iloc[:,11]=='high') ])/len(trainingSecond[trainingSecond.iloc[:,11]=='high'])
    propHighHSTMedium=len(trainingSecond[(trainingSecond.iloc[:,column]=='medium') & (trainingSecond.iloc[:,11]=='high') ])/len(trainingSecond[trainingSecond.iloc[:,11]=='high'])
    propHighHSTLow=len(trainingSecond[(trainingSecond.iloc[:,column]=='low') & (trainingSecond.iloc[:,11]=='high') ])/len(trainingSecond[trainingSecond.iloc[:,11]=='high'])
    
    if len(trainingSecond[(trainingSecond.iloc[:,column]=='high')&(trainingSecond.iloc[:,11]=='high')])==0:
        phighHSTHighHome=0
        phighHSTHighAway=0
    else:
        phighHSTHighHome= len(trainingSecond[(trainingSecond.iloc[:,column]=='high') & (trainingSecond.iloc[:,11]=='high')& (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='high')&(trainingSecond.iloc[:,11]=='high')])
        phighHSTHighAway= len(trainingSecond[(trainingSecond.iloc[:,column]=='high') & (trainingSecond.iloc[:,11]=='high')& (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='high')&(trainingSecond.iloc[:,11]=='high')])
        
    if len(trainingSecond[(trainingSecond.iloc[:,column]=='medium')&(trainingSecond.iloc[:,11]=='high')])==0:
        phighHSTMediumHome=0
        phighHSTMediumAway=0
    else:
        phighHSTMediumHome= len(trainingSecond[(trainingSecond.iloc[:,column]=='medium') & (trainingSecond.iloc[:,11]=='high')& (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='medium')&(trainingSecond.iloc[:,11]=='high')])
        phighHSTMediumAway= len(trainingSecond[(trainingSecond.iloc[:,column]=='medium') & (trainingSecond.iloc[:,11]=='high')& (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='medium')&(trainingSecond.iloc[:,11]=='high')])
    
    if len(trainingSecond[(trainingSecond.iloc[:,column]=='low')&(trainingSecond.iloc[:,11]=='high')])==0:
        phighHSTLowHome=0
        phighHSTLowAway=0
    else: 
        phighHSTLowHome= len(trainingSecond[(trainingSecond.iloc[:,column]=='low') & (trainingSecond.iloc[:,11]=='high')& (trainingSecond.iloc[:,12]=='H')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='low')&(trainingSecond.iloc[:,11]=='high')])
        phighHSTLowAway= len(trainingSecond[(trainingSecond.iloc[:,column]=='low') & (trainingSecond.iloc[:,11]=='high')& (trainingSecond.iloc[:,12]=='A')])/len(trainingSecond[(trainingSecond.iloc[:,column]=='low')&(trainingSecond.iloc[:,11]=='high')])
        
    GainSecond=EntropyHST-propHighHSTHigh*Entropy(phighHSTHighHome,phighHSTHighAway)-propHighHSTMedium*Entropy(phighHSTMediumHome,phighHSTMediumAway)-propHighHSTLow*Entropy(phighHSTLowHome,phighHSTLowAway)
    HighHSTGain.append(GainSecond)

print('The attribute with the highest gain for second Node with HST(the root node) is high is '+trainingSecond.columns.values[np.argmax(HighHSTGain)]+' and its value is '+str(np.max(HighHSTGain)))
print('Number of times home win in case of high HST and low HY = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(HighHSTGain)]=='low') & (trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='H')])))
print('Number of times away win in case of high HST and low HY = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(HighHSTGain)]=='low') & (trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='A')])))
print('Number of times home win in case of high HST and medium HY = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(HighHSTGain)]=='medium') & (trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='H')])))
print('Number of times away win in case of high HST and medium HY = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(HighHSTGain)]=='medium') & (trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='A')])))
print('Number of times home win in case of high HST and high HY = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(HighHSTGain)]=='high') & (trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='H')])))
print('Number of times away win in case of high HST and high HY = '+str(len(trainingSecond[(trainingSecond.iloc[:,np.argmax(HighHSTGain)]=='high') & (trainingSecond.iloc[:,11]=='high') & (trainingSecond.iloc[:,12]=='A')])))


###################################################################################
#calcualating the gain for all features to identify the third level splitting nodes
###################################################################################


#################################################################################################
#The below function has three inputs the data frame itself, and the level of third
#level branch(for example low AST) and the last input shows the level of HST(high or low)
#the output of the function are some prints showing the attribute that has maximum gain in
#addition to number of H/A match winner in case of low, medium and high values of this attribute
#################################################################################################

def ThirdLevelGain(dataframe,third_level,second_level):
    ASTLowHome=len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level) & (dataframe.iloc[:,12]=='H')])/len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
    ASTLowAway=len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level) & (dataframe.iloc[:,12]=='A')])/len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
    EntropyAST=Entropy(ASTLowHome,ASTLowAway)
    
    LowASTGain=[]
    for column in range(dataframe.shape[1]-4):
        proplowASTHigh=len(dataframe[(dataframe.iloc[:,column]=='high') &(dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level) ])/len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
        proplowASTMedium=len(dataframe[(dataframe.iloc[:,column]=='medium') &(dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level) ])/len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
        proplowASTLow=len(dataframe[(dataframe.iloc[:,column]=='low') &(dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level) ])/len(dataframe[(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
        
        if len(dataframe[(dataframe.iloc[:,column]=='high')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])==0:
            plowASTHighHome=0
            plowASTHighAway=0
        else:
            plowASTHighHome= len(dataframe[(dataframe.iloc[:,column]=='high')& (dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,12]=='H')])/len(dataframe[(dataframe.iloc[:,column]=='high')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
            plowASTHighAway= len(dataframe[(dataframe.iloc[:,column]=='high')& (dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,12]=='A')])/len(dataframe[(dataframe.iloc[:,column]=='high')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
            
        if len(dataframe[(dataframe.iloc[:,column]=='medium')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])==0:
            plowASTMediumHome=0
            plowASTMediumAway=0
        else:
            plowASTMediumHome= len(dataframe[(dataframe.iloc[:,column]=='medium')& (dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,12]=='H')])/len(dataframe[(dataframe.iloc[:,column]=='medium')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
            plowASTMediumAway= len(dataframe[(dataframe.iloc[:,column]=='medium')& (dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,12]=='A')])/len(dataframe[(dataframe.iloc[:,column]=='medium')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
        
        if len(dataframe[(dataframe.iloc[:,column]=='low')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])==0:
            plowASTLowHome=0
            plowASTLowAway=0
        else: 
            plowASTLowHome= len(dataframe[(dataframe.iloc[:,column]=='low')& (dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,12]=='H')])/len(dataframe[(dataframe.iloc[:,column]=='low')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
            plowASTLowAway= len(dataframe[(dataframe.iloc[:,column]=='low')& (dataframe.iloc[:,10]==third_level)& (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,12]=='A')])/len(dataframe[(dataframe.iloc[:,column]=='low')&(dataframe.iloc[:,10]==third_level)&(dataframe.iloc[:,11]==second_level)])
            
        GainThird=EntropyAST-proplowASTHigh*Entropy(plowASTHighHome,plowASTHighAway)-proplowASTMedium*Entropy(plowASTMediumHome,plowASTMediumAway)-proplowASTLow*Entropy(plowASTLowHome,plowASTLowAway)
        #print(GainThird)
        LowASTGain.append(GainThird)
    print('The attribute with the highest gain for third node Node with HST is '+second_level+' and '+dataframe.columns[10]+ ' is '+third_level+ ' is ' +dataframe.columns.values[np.argmax(LowASTGain)]+' and its value is '+str(np.max(LowASTGain)))
    print('Number of times home win when attribute is low  '+str(len(dataframe[(dataframe.iloc[:,np.argmax(LowASTGain)]=='low') & (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,10]==third_level) & (dataframe.iloc[:,12]=='H')])))
    print('Number of times away win when attribute is low '+str(len(dataframe[(dataframe.iloc[:,np.argmax(LowASTGain)]=='low') & (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,10]==third_level) & (dataframe.iloc[:,12]=='A')])))
    print('Number of times home win when attribute is medium  '+str(len(dataframe[(dataframe.iloc[:,np.argmax(LowASTGain)]=='medium') & (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,10]==third_level) & (dataframe.iloc[:,12]=='H')])))
    print('Number of times away win when attribute is medium '+str(len(dataframe[(dataframe.iloc[:,np.argmax(LowASTGain)]=='medium') & (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,10]==third_level) & (dataframe.iloc[:,12]=='A')])))
    print('Number of times home win when attribute is high  '+str(len(dataframe[(dataframe.iloc[:,np.argmax(LowASTGain)]=='high') & (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,10]==third_level) & (dataframe.iloc[:,12]=='H')])))
    print('Number of times away win when attribute is high '+str(len(dataframe[(dataframe.iloc[:,np.argmax(LowASTGain)]=='high') & (dataframe.iloc[:,11]==second_level)& (dataframe.iloc[:,10]==third_level) & (dataframe.iloc[:,12]=='A')])))
 

#####################################
#calcualting gains for low HST branch
#####################################

#Firstly we copied the dataframe from the last level and we will change indices
trainingThirdLow = trainingSecond.copy()

new_index_Third_low=['HS', 'AS', 'HF', 'AF', 'HC', 'AC', 'AY', 'HR','AR','HY', 'AST','HST','FTR']
trainingThirdLow=trainingThirdLow.reindex(new_index_Third_low,axis=1)

# we will call the function 2 times first time putting AST as low and the second time is medium noting 
#that HST is fixed to be low in this branch

ThirdLevelGain(trainingThirdLow,'low','low')
ThirdLevelGain(trainingThirdLow,'medium','low')

#####################################
#calcualting gains for high HST branch
#####################################

#Firstly we copied the dataframe from the last level and we will change indices
trainingThirdHigh = trainingSecond.copy()

new_index_Third_high=['HS', 'AS', 'HF', 'AF', 'HC', 'AC', 'AY', 'HR','AR','AST','HY','HST','FTR']
trainingThirdHigh=trainingThirdHigh.reindex(new_index_Third_high,axis=1)

# we will call the function 3 times first time putting HY as low and the second time is medium
# and the third time we put HY as high noting that HST is fixed to be high in this branch

ThirdLevelGain(trainingThirdHigh,'low','high')
ThirdLevelGain(trainingThirdHigh,'medium','high')
ThirdLevelGain(trainingThirdHigh,'high','high')

#####################
#Reading testing data
#####################
os.chdir("E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 3")

# read all columns except for the first two which we will not use
usedCols=list(range(2,15))
testing=pd.read_excel('Liverpool.xlsx',usecols=usedCols)

# take a copy of the original dataFrame and discretize columns values for root node calcualtion
testingDisc = testing.copy()

#Discretinzing all columns except for the last column that shows match result noting that we will
#Discetize based on the values calcualted in training data

for column in range(testing.shape[1]-1):
    #print(column)
    Min=np.min(training.iloc[:,column])
    Max=np.max(training.iloc[:,column])
    Range=np.max(training.iloc[:,column])-np.min(training.iloc[:,column])
    First=Min+(Range/3)
    Second=Range*(2/3)+Min
    testingDisc.iloc[:,column]=pd.cut(testing.iloc[:,column],bins=[Min,First,Second,Max],labels=["low","medium","high"],include_lowest=True)

#We will change Descitization for HST as it should contains only 2 levels
    
colMin=np.min(training.iloc[:,2])
colMean=np.mean(training.iloc[:,2])
colMax=np.max(training.iloc[:,2])
testingDisc.iloc[:,2]=pd.cut(testing.iloc[:,2],bins=[colMin,colMean,colMax],labels=["low","high"],include_lowest=True)

#######################################################################################
#Building 3-level decision tree function based on the values granted from training data
#######################################################################################
FTRtotal=[]
for match in range(len(testingDisc)):
    FTRtesting=[]
    if (testingDisc.loc[match,"HST"]=='high') & (testingDisc.loc[match,"HY"]=='low'):
        FTRtesting='H'
    elif (testingDisc.loc[match,"HST"]=='high') & (testingDisc.loc[match,"HY"]=='medium') & (testingDisc.loc[match,"HF"]!='high'):
        FTRtesting='H'
    elif (testingDisc.loc[match,"HST"]=='high') & (testingDisc.loc[match,"HY"]=='high') & (testingDisc.loc[match,"AY"]=='low'):
        FTRtesting='H'
    elif (testingDisc.loc[match,"HST"]=='low') & (testingDisc.loc[match,"AST"]=='low') & (testingDisc.loc[match,"AY"]!='medium'):
        FTRtesting='H'
    else:
        FTRtesting='A'
    FTRtotal.append(FTRtesting)


##############################################################
#Measuring the algorithm accuracy and getting confusion matrix
##############################################################
#calculating Accuracy
TruePrediction=0
for match in range(len(FTRtotal)):
    if FTRtotal[match]==testing.loc[match,'FTR']:
        TruePrediction+=1
 
       
Accuracy= (TruePrediction/len(FTRtotal))*100
print("Accuracy is "+str(Accuracy)+"%"  )

#Getting confusion matrix      
conf_matrix=confusion_matrix(testing.loc[:,'FTR'],FTRtotal,labels=['H','A'])

#################################
#Saving confusion matrix as image
#################################

fig = plt.figure()
ax = fig.add_subplot(111)
cax=ax.matshow(conf_matrix,cmap=plt.cm.Blues)
fig.colorbar(cax)
plt.title('Accuracy = ' +str(Accuracy) +'%')
plt.savefig('confusion_matrix.jpg')


