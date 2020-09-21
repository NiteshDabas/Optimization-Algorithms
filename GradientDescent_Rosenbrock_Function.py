######################################################
### Step Gradient Algorithm
### MINIMIZE The Banana (Rosenbrock) Function: 100*((y-x**2)**2)+(1-x)**2
### Author: NITESH DABAS
######################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

np.seterr(all="ignore")  #to supress all warnings
#np.seterr(all="print") 

#calculate te derivative for Rosenbrock Function - 100*((y-x**2)**2)+(1-x)**2
def DerivativeRosenbrock (point):
    dx = (-2*(1 - point[0])) - 400*point[0]*(point[1] - (point[0]**2))
    dy = 200*(point[1] - (point[0]**2))
    return dx, dy

#Minimize the function
def minimize():
    #set variables
    cur_f = 0    #current objective function value at defined point
    iters = 0    #Iteration variable
    precision = 0.0000001 #alarm to stop algo
    iterations = 100000   #set max number of iterations
    previous_step_size = 1   
    ai = []     #list for observation points
    fnval = []  #list for objective function value wit each iteration
    Error_Overflow =False # scalar exception error
    Plot=True   #Variable to differentiate plots without error or with error!
    lstep = 0.0002   # set the step for gradient algo
    
    # initialize a point with some random value b/w [-5,5]
    a = np.array([np.random.randint(-5,5), np.random.randint(-5,5)])
    initial_point = a
    #when starting point itself is a minimum
    if a[0]==1.00 and a[1]==1.00:
        print("\tStarting point itself is a minima.\n")
        Plot=False
        
    #loop to minimize objective function
    while previous_step_size > precision and iters < iterations:
        prev_f = cur_f
        #objective function 
        f = ((1 - a[0])**2) + (100*((a[1] - a[0]**2)**2))
        #raise OverflowError() true
        if abs(f)==float('inf'): 
            Error_Overflow=True
            f = fnval
            break
        #append the point and its obj function to ai/fnval lists
        ai.append([a,f])
        fnval.append(f) #mainly for plot       
        #reassign objective function value at point a
        cur_f = f
        #compute its derrivative on point a
        fi = np.array(DerivativeRosenbrock(a))
        #reassign step value at point a
        a = a - lstep*fi
        previous_step_size = abs(cur_f - prev_f)
        #increase iteration by 1.
        iters=iters+1
    
    # convert ai/fnval into a numpy array and do operations
    ai = np.array(ai)
    minFnVal=min(np.array(fnval))
    minIndex= fnval.index(minFnVal)
    #print the findings
    #print(f'\n\tThe minimum is: {minFnVal} at point: {ai[minIndex,0]} after iterations: {iters}.')
    print(f'\t\tStarting Point:   {[initial_point[0],initial_point[1]]}\n\t\tFinal Point:      {ai[minIndex,0]}\
          \n\t\tMinimum Value:    {round(minFnVal,5)}\n\t\tTotal Iterations: {iters}')
    
    
    # Plot the decline in function value
    if(Plot == True):
        fig1,ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        #Plot with red line if there were an overflow while calculating function value
        #Print appropriate message
        if Error_Overflow:
            ax.plot(range(iters),fnval,color="r")
            print("overflow encountered. Function is NOT going to minimize with starting point:{} and step-size:0.002".format(ai[0,0]))
        else:        
            ax.plot(range(iters),fnval,color="b")    
        #loop for formatting x/y axis ticks
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        #draw a red line for function decline intersection at function value 0.001 for ease in understanding 
        plt.axhline(y=minFnVal,color="r",linestyle='--')
        #Title and label the graph        
        plt.title("Objective Function Value vs Iteartions\n",fontsize=16, fontweight='bold')
        plt.xlabel("Iterations",fontsize=16, fontweight='bold')
        plt.ylabel("Objective Function value",fontsize=16, fontweight='bold') 
        plt.show()

minimize()   
