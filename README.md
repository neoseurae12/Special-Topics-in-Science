```python
import numpy as np
```

# The Dynamics of Internal Migration in S.Korea: A Markov Chain Analysis


<img src="https://velog.velcdn.com/images/neoseurae12/post/44c092b1-c747-415e-8d2d-dd0762dccc6a/image.jpeg" width="500px">


```python
state = {
    0: "Seoul",
    1: "Metropolitan",
    2: "Rural"
}
state
```




    {0: 'Seoul', 1: 'Metropolitan', 2: 'Rural'}



## Transition Matrix
<img src="https://velog.velcdn.com/images/neoseurae12/post/aeeaf770-15e2-4f57-aa64-4f5a184cf0d8/image.jpeg" width="400px">


```python
A = np.array(
    [[0.646, 0.246, 0.108],
    [0.121, 0.703, 0.176],
    [0.055, 0.115, 0.830]])
A
```




    array([[0.646, 0.246, 0.108],
          [0.121, 0.703, 0.176],
          [0.055, 0.115, 0.830]])



## Random Walk on Markov Chain


```python
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
```

    Rural ---> Rural ---> Rural ---> Rural ---> Rural ---> Metropolitan ---> Rural ---> Rural ---> Rural ---> Rural ---> Seoul ---> Seoul ---> stop


## Approach 1: Monte Carlo


```python
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
```

    π = 
     [0.189996 0.339892 0.470113]


## Approach 2: Repeated Matrix Multiplication


```python
steps = 10**3
A_n = A

i = 0
while i < steps:
    A_n = np.matmul(A_n, A)
    i += 1
    
print("A^n = \n", A_n, "\n")
print("π = ", A_n[0])
```

    A^n = 
     [[0.18922571 0.33929264 0.47148165]
     [0.18922571 0.33929264 0.47148165]
     [0.18922571 0.33929264 0.47148165]] 

    π =  [0.18922571 0.33929264 0.47148165]


## Approach 3: Finding Left Eigen Vectors


```python
import scipy.linalg
values, left = scipy.linalg.eig(A, right=False, left=True)

print("left eigen vectors = \n", left, "\n")
print("eigen values = \n", values)
```

    left eigen vectors = 
     [[ 0.30973994  0.55586939 -0.34067498]
     [ 0.55538163 -0.79586994 -0.47227859]
     [ 0.7717593   0.24000055  0.81295356]] 
    
    eigen values = 
     [1.        +0.j 0.49650403+0.j 0.68249597+0.j]



```python
pi = left[:,0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
pi_normalized
```




    [0.1892257071724362, 0.3392926399019156, 0.47148165292564814]
