#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # The Dynamics of Internal Migration in S.Korea: A Markov Chain Analysis
# 

# <img src="https://velog.velcdn.com/images/neoseurae12/post/0e578941-88f2-4dfc-a466-2ec0d3741f74/image.jpeg" width="500px">

# In[2]:


state = {
    0: "Seoul",
    1: "Metropolitan",
    2: "Rural"
}
state


# ## Transition Matrix
# <img src="https://velog.velcdn.com/images/neoseurae12/post/aeeaf770-15e2-4f57-aa64-4f5a184cf0d8/image.jpeg" width="400px">

# In[4]:


A = np.array(
    [[0.646, 0.246, 0.108],
    [0.121, 0.703, 0.176],
    [0.055, 0.115, 0.830]])
A


# ## Random Walk on Markov Chain

# In[9]:


n = 12
start_state = 2
print(state[start_state], "--->", end=" ")
prev_state = start_state

while n - 1:
    curr_state = np.random.choice([0, 1, 2], p=A[prev_state])
    print(state[curr_state], "--->", end=" ")
    prev_state = curr_state
    n -= 1
print("stop")


# ## Approach 1: Monte Carlo

# In[29]:


steps = 10**6
start_state = 0
pi = np.array([0, 0, 0])
pi[start_state] = 1
prev_state = start_state

i = 0
while i < steps:
    curr_state = np.random.choice([0, 1, 2], p=A[prev_state])
    pi[curr_state] += 1
    prev_state = curr_state
    i += 1

print("π = \n", pi/steps)


# ## Approach 2: Repeated Matrix Multiplication

# In[28]:


steps = 10**3
A_n = A

i = 0
while i < steps:
    A_n = np.matmul(A_n, A)
    i += 1
    
print("A^n = \n", A_n, "\n")
print("π = ", A_n[0])


# ## Approach 3: Finding Left Eigen Vectors

# In[30]:


import scipy.linalg
values, left = scipy.linalg.eig(A, right=False, left=True)

print("left eigen vectors = \n", left, "\n")
print("eigen values = \n", values)


# In[31]:


pi = left[:,0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
pi_normalized

