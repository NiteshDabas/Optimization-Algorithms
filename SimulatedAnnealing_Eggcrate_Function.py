######################################################
### SIMULATED ANNEALING - METAHEURISTIC
### MINIMIZE The Eggcrate Function: x**2+y**2+25*(np.sin(x)**2+np.sin(y)**2)
### Author: NITESH DABAS
######################################################

import math
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="ignore") # ignores all warning

def minimize():
    x = np.random.randint(-2*(math.pi), 2*(math.pi)) # initial x selected randomly b/w [-2pi,2pi] 
    y = np.random.randint(-2*(math.pi), 2*(math.pi)) # initial y selected randomly b/w [-2pi,2pi]
    initial_point =[x,y]

    T0 = 1000 # set initial temperature
    temp_for_plot = T0
    M = 10000 #Max Iterations-How many times you want to decrtease the temperature
    N = 20 #neighbourhood search
    alpha = 0.85 #cooling parameter
    k = 4 #step multiplier
    iCount = 0
    precision = 0.01 #alarm to stop algo when objective function chnage is less tan precision
    temp = [] # empty list for varying temperatures
    obj_val = [] # empty list for objective function values
            
    for i in range(M): #how many iterations?     
        for j in range(N): #neighbourhood seach count for eac iteration   
            
            #For variable x
            rand_num_x_1 = np.random.rand() # increase or decrease x value?
            rand_num_x_2 = np.random.rand() # by how much?
            if rand_num_x_1 >= 0.5: # greater than 0.5, we increase
                step_size_x = k * rand_num_x_2 # step-size
            else:
                step_size_x = -k * rand_num_x_2 # less than 0.5, we decrease  
            
            #For variable y
            rand_num_y_1 = np.random.rand() # increase or decrease y value?
            rand_num_y_2 = np.random.rand() # by how much?
            if rand_num_y_1 >= 0.5: # greater than 0.5, we increase
                step_size_y = k * rand_num_y_2 # step-size
            else:
                step_size_y = -k * rand_num_y_2 # less than 0.5, we decrease
                     
            #new x and y coordinates for potentoial next step
            x_temporary = x + step_size_x
            y_temporary = y + step_size_y
            #new objective function value
            obj_val_possible = x_temporary**2+y_temporary**2+25*(np.sin(x_temporary)**2+np.sin(y_temporary)**2)
            
            # where we are currently
            obj_val_current = x**2+y**2+25*(np.sin(x)**2+np.sin(y)**2)
            
            #should we take a bad step?
            rand_num = np.random.rand()
            formula = 1/(np.exp((obj_val_possible - obj_val_current)/T0))            
            if obj_val_possible <= obj_val_current: # it's a better solution anyhow
                x = x_temporary
                y = y_temporary
            elif rand_num <= formula: # it's a bad solution and we take it
                x = x_temporary
                y = y_temporary
            else: # it's a bad solution and we take reject it
                x = x
                y = y
        iCount=iCount+1     
        temp.append(T0) #build on this list of temperatures
        obj_val.append(obj_val_current) #build on this list of objective function values          
        T0 = alpha*T0 #lets cool-off b4 next step. lets decrease te temperature. 
        if round(obj_val_current,3) < precision:
            break

    print(f'\t\tStarting Point:   {[initial_point[0],initial_point[1]]}\n\t\tFinal Point:      {[round(x,5),round(y,5)]}\
          \n\t\tMinimum Value:    {round(min(np.array(obj_val)),5)}\n\t\tTotal Iterations: {iCount}.')
   
    #plot the objective functio variation as temperature declines
    plt.plot(temp,obj_val)
    plt.axhline(y=min(np.array(obj_val)),color="r",linestyle='--')
    plt.title("Objective Function Variation as temperature declines\n",fontsize=16, fontweight='bold')
    plt.xlabel("Temperature\nEggcrate Function",fontsize=16, fontweight='bold')
    plt.ylabel("Objective Function value",fontsize=16, fontweight='bold') 
    plt.xlim(temp_for_plot,0)
    plt.xticks(np.arange(min(temp),max(temp),100),fontweight='bold')
    plt.yticks(fontweight='bold')
    #finally plot
    plt.show()
    
minimize()
