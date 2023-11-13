# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:25:21 2023

@author: hlin_
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def assign_mol(starting_state):
    '''
    take in the site number and create a new empty array
    
    Parameter
    ---------
    starting_state(int): starting site 
    
    Returns
    -------
    traj_arr(arr) : trajectory array that stores the trajectory of the atom
    time(arr) : an array to keep track of the time taken for each transition
    '''
    traj_arr = np.array([starting_state])
    time = np.array([0])
    return traj_arr,time

def rate_catalogue(site_no):
    '''
    return the rate catalogue of the given state
    
    Parameter
    ---------
    site_no(int): the current site the walker is on
    
    Returns
    -------
    k(arr) : the rate array 
    '''
    # rate parameter
    
    #define concentration
    concentration = 0.05
    #estimated rate
    k_dif = 3000000 #concentration dependent diffusion rate. 3.0x10^6 M^-1 s^-1,
    k_f = 30 #concentration dependent forward rate for cartwheeling 3x10^1
    k_dif_c = k_dif*concentration 
    k_front = k_f*concentration
    k_out = 0.2
    
    if (site_no == 0): 
        k = np.array([0,k_out,2,0])
    elif (site_no == 1): 
        k = np.array([0,k_out,2,0.5])
    elif (site_no == 2): 
        k = np.array([0,k_out,2,0.5])
    elif (site_no == 3):
        k = np.array([0,k_out,2,0.5])
    elif (site_no == 4): 
        k = np.array([0,k_out,2,0.5])
    elif (site_no == 5):
        k = np.array([0,k_out,0,0])
    return k 

def state_update(traj_arr,time,current_state, isempty):
    '''
    generate a random number and decide the state transition
    update the state, trajectory and time after a state transition.
    
    Parameters
    ----------
    traj_arr(arr) :
    time (arr) :
    current_state(int) :
    isempty(boolean) :
    
    Returns
    -------
    traj_arr(arr) : 
    time(float) :
    isempty(bool) :
    current_state(int) :
    '''
    # call the rate catalogue
    k = rate_catalogue(current_state) 
    # determine the state transition and the time taken needed
    next_state,time = determine_transition(k,current_state,time) 
    print(traj_arr)
    # update the trajectory 
    traj_arr = np.append(traj_arr,next_state)
    current_state = next_state
    #if the molecule fly away, then update isempty
    if next_state == -1:
        isempty = 1
    else:
        isempty = 0
    return traj_arr, time, isempty,current_state

def determine_transition(k,current_state, time):
    '''
    take in the rate catalogue. Generate a random number. 
    Determine which transition to take
    Calculate the time needed for the transition
    
    Parameters
    ----------
    k(arr) : an array of possible state transition and its rate
    current_state(int) :
    time(arr) : an array for time for each transition
    Returns
    -------
    next_state(int) : which state to transit
    time(float) : updated time array
    '''
    ran1 = random.random()
        #check the state transition
    temp = 0
    k_tot = np.sum(k)
    for index, item in enumerate(k):
        temp += item
        if (ran1*k_tot < temp):
            transition = index 
            break
    #current rate catalogue only allows transition forward, backward or fly away
    
    if transition == 1: #fly away
        next_state = -1
    elif transition == 2: #forward
        next_state = current_state + 1
    elif transition == 3: #backward
        next_state = current_state - 1
    print('Transition from state: ', current_state, 'to state: ', next_state)
    time = calculate_time(time,k) #calculate the transition time for the chosen transition
    return next_state, time

def calculate_time(time, k):
    '''
    calculate the transition time

    Parameters
    ----------
    time(arr) : previous time arr
    k(arr) : TYPE
        DESCRIPTION.
    Returns
    -------
    time(arr) : accumulated transition time

    '''
    k_tot = np.sum(k)
    ran2 = random.random()
    time = np.append(time, time[-1]+math.log(1/ran2)/k_tot)
    return time

def plot_trajectory(history_arr,time_arr,no):
    '''
    Plot the first no-th step-wise trajectory of the molecule based on the transition. 
    
    Parameters
    ----------
    history_arr(arr) :
    time_arr(arr) :
    no(int) : number of trajectories
    
    Returns
    -------
    '''
    fig,ax = plt.subplots()
    for index in range(no):
        ax.step(time_arr[index][:-1],history_arr[index][:-1], where='post')
        ax.plot(time_arr[index][:-1],history_arr[index][:-1], 'o--', color='grey', alpha=0.3)
    ax.set_xlabel('Time, s.u.')
    ax.set_ylabel('Site')
    title_string = 'The first '+ str(no) + ' trajectories plot'
    ax.set_title(title_string)
    
def occupancy(site, history_arr,time_arr, time_interval, time_until):
    '''
    Return the occupancy(Define as percentage of molecule in the site) at a fixed time interval

    Parameters
    ----------
    site : TYPE
        DESCRIPTION.
    history_arr : TYPE
        DESCRIPTION.
    time_arr : TYPE
        DESCRIPTION.

    Returns
    -------
    occup(arr) : 

    '''
    track_no = len(history_arr)
    occup = [0 for x in range(math.ceil(time_until/time_interval))]
    for index1,traj_arr in enumerate(history_arr):
        for index2, item in enumerate(traj_arr):
            #check if the trajectory reaches the site of interest
            if item == site:
                # check the time it stays at the site of interest
                time_start = time_arr[index1][index2]
                idletime = time_arr[index1][index2 + 1] - time_start
                #print('The molecule is in site', site, 'at this time:', time_start, 'for', round(idletime,2))
                # update the occupancy array accordingly
                occup = binning_occupancy(time_start, idletime,time_interval, time_until,occup)
    occup = [1- x/track_no for x in occup]
    x = [i*time_interval for i in range(len(occup))]
    return occup,x
            
def binning_occupancy(time_start,idletime,time_interval, time_until,occup):
    '''
    from the idle time, calculate how long the molecule stays in the site, 
    and return the right occupancy signal
    
    Parameters
    ----------
    time_start(float) :
    idletime(float) : 
    time_interval(float) :
    time_until(float) :
    occupancy(arr) :
        
    Returns
    -------
    occupancy(arr) :
    '''
    #calculate how many bins in the occupancy array
    bin_no = len(occup)
    accum_time = 0
    time_end = time_start+idletime
    for i in range(bin_no):
       # print('Current accumulated time:', accum_time,'and the targetted time', time_start)
       # print('Accum time is ', round(accum_time,2), time_start > accum_time)
       # print('And the itnerval is' ,accum_time <= time_end)
        if (accum_time >= time_start) and (accum_time < time_end): 
            occup[i] += 1
            #print('Updating the occupance of chosen site at time ',accum_time,'from',occup[i]-1 ,'to', occup[i])
        accum_time += time_interval
    return occup

########### Start of the simulation
    #create 6 site lattices
step = 50000
lattice_size = 6
#history 
traj_arr = np.array([])
history_arr = []
time_arr = []
starting_site = 1
isempty = 1
current_state = 0
time = 0
for i in range(step):
    #randomly pick a site, assign a molecule
    if isempty == 1: #if track is empty, assign a particle on a track, reset rate
        print('Assigning new molecule ...')
        traj_arr,time = assign_mol(starting_site)
        current_state = starting_site
    traj_arr,time, isempty, current_state = state_update(traj_arr,time,current_state,isempty)
    print(isempty)
    #if molecule flies away, store the trajectory array into a history array. reset the trajectory array
    if isempty == 1:
        history_arr.append(traj_arr)
        time_arr.append(time)
        print('Molecule fly away,took', round(np.sum(time),2),'s.u. to fly away,  resetting ...')


##############   Analysis   #############
#plot all the trajectory
no = 10
plot_trajectory(history_arr,time_arr,no)

time_interval = 0.1
time_until = 9  #collect the statistic from time 0 to time_until
site1 = 2
site2 = 3
site3 = 5
occup1,x1 = occupancy(site1, history_arr,time_arr, time_interval, time_until)
occup2,x2 = occupancy(site2, history_arr,time_arr, time_interval, time_until)
occup3,x3 = occupancy(site3, history_arr,time_arr, time_interval, time_until)
fig2,ax2 = plt.subplots()
ax2.plot(x1,occup1,'-o', label = site1)
ax2.plot(x2,occup2,'-o', label = site2)
ax2.plot(x3,occup3,'-o', label = site3)
ax2.set_xlabel('Time, s.u.')
ax2.set_ylabel('1 - Occupancy, %')
ax2.legend()