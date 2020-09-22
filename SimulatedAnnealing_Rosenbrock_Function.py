######################################################
### SIMULATED ANNEALING - METAHEURISTIC
### MINIMIZE The Banana (Rosenbrock) Function: 100*((y-x**2)**2)+(1-x)**2
### Author: NITESH DABAS
######################################################

import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide="ignore") # ignores the warning of zero division

def minimize():
    x = np.random.randint(-5,5) # initial x selected randomly b/w [-5,5] 
    y = np.random.randint(-5,5) # initial y selected randomly b/w [-5,5] 
    initial_point =[x,y]
    
    T0 = 1000 # set initial temperature
    temp_for_plot = T0
    M = 10000 #Max Iterations
    N = 15 #neighbourhood searc
    alpha = 0.85 #cooling parameter
    k = 0.1 #step multiplier
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
            obj_val_possible = 100*(y_temporary-x_temporary**2)**2 + (1-x_temporary)**2
            
            # where we are currently
            obj_val_current = 100*(y-x**2)**2+(1-x)**2  
            
            #should we take a bad step?
            rand_num = np.random.rand()
            formula = 1/(np.exp((obj_val_possible - obj_val_current)/T0))            
            if obj_val_possible <= obj_val_current: # it's better a better solution anyhow
                x = x_temporary
                y = y_temporary
            elif rand_num <= formula: # it's better a bad solution and we take it
                x = x_temporary
                y = y_temporary
            else: # it's better a bad solution and we take reject it
                x = x
                y = y
                
        iCount=iCount+1     
        temp.append(T0) #build on this list of temperatures
        obj_val.append(obj_val_current) #build on this list of objective function values          
        T0 = alpha*T0 #lets cool-off b4 next step. lets decrease te temperature.
        if round(obj_val_current,3) < precision:
            break
    
    #print(f'\n\tThe minimum is: {min(np.array(obj_val))} at point: {x,y} after iterations: {iCount}.')
    print(f'\t\tStarting Point:   {[initial_point[0],initial_point[1]]}\n\t\tFinal Point:      {[round(x,5),round(y,5)]}\
          \n\t\tMinimum Value:    {round(min(np.array(obj_val)),5)}\n\t\tTotal Iterations: {iCount}.')
   
    
    #plot the objective functio variation as temperature declines
    plt.plot(temp,obj_val)
    plt.axhline(y=min(np.array(obj_val)),color="r",linestyle='--')
    plt.title("Objective Function Variation as temperature declines\n",fontsize=16, fontweight='bold')
    plt.xlabel("Temperature\nRossenbrock Function",fontsize=16, fontweight='bold')
    plt.ylabel("Objective Function value",fontsize=16, fontweight='bold') 
    plt.xlim(temp_for_plot,0)
    plt.xticks(np.arange(min(temp),max(temp),100),fontweight='bold')
    plt.yticks(fontweight='bold')
    #finally plot
    plt.show()
    
minimize()
