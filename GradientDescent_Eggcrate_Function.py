######################################################
### Step Gradient Algorithm
### MINIMIZE The Eggcrate Function: x**2+y**2+25*(np.sin(x)**2+np.sin(y)**2)
### Author: NITESH DABAS
######################################################

import math
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="ignore")  #to supress all warnings
#np.seterr(all="print") 

#calculate te derivative for Rosenbrock Function - 100*((y-x**2)**2)+(1-x)**2
def DerivativeEggcrate(point):
    dx = (2*point[0] + 25*(np.sin(2*point[0])))
    dy = (2*point[1] + 25*(np.sin(2*point[1])))
    return dx, dy

#Minimize the function
def minimize():
    #set variables
    cur_f = 0    #current objective function value at defined point
    iters = 0    #Iteration variable
    precision = 0.01 #alarm to stop algo
    iterations = 10000   #set max number of iterations
    previous_step_size = 1   
    ai = []     #list for observation points
    fnval = []  #list for objective function value wit each iteration
    lstep = 0.1   # set the step for gradient algo
    # initialize a point with some random value b/w [-2pi,2pi]
    a = np.array([np.random.randint(-2*(math.pi), 2*(math.pi)), np.random.randint(-2*(math.pi), 2*(math.pi))])
    initial_point = a
    #when starting point itself is a minimum
    if a[0]==0.00 and a[1]==0.00:
        print("\t\tStarting point itself is a minimum point.\n")
    #loop to minimize objective function
    while previous_step_size > precision and iters < iterations:
        prev_f = cur_f
        #objective function 
        f = a[0]**2+a[1]**2+25*(np.sin(a[0])**2+np.sin(a[1])**2)
        #append the point and its obj function to ai/fnval lists
        ai.append([a,f])
        fnval.append(f) #mainly for plot       
        #reassign objective function value at point a
        cur_f = f
        #compute its derrivative on point a
        fi = np.array(DerivativeEggcrate(a))
        #reassign step value at point a
        a = a - np.dot(lstep,fi)
        previous_step_size = abs(cur_f - prev_f)
        #increase iteration by 1.
        iters=iters+1
    
    # convert ai/fnval into a numpy array and do operations
    ai = np.array(ai)
    minFnVal=min(np.array(fnval))
    minIndex= fnval.index(minFnVal)
    #print the findings
    print(f'\t\tStarting Point:   {[initial_point[0],initial_point[1]]}\n\t\tFinal Point:      {ai[minIndex,0]}\
          \n\t\tMinimum Value:    {round(minFnVal,5)}\n\t\tTotal Iterations: {iters}')
    
    # Plot the decline in function value
    plt.plot(range(iters),fnval)
    plt.axhline(y=minFnVal,color="r",linestyle='--')
    plt.title("Objective Function Value vs Iteartions\n",fontsize=16, fontweight='bold')
    plt.xlabel("Iterations",fontsize=16, fontweight='bold')
    plt.ylabel("Objective Function value",fontsize=16, fontweight='bold') 
    plt.show()
    
minimize()   
