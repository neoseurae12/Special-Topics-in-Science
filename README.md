```python
import numpy as np
```

# The Dynamics of Internal Migration in S.Korea: A Markov Chain Analysis


<img src="https://velog.velcdn.com/images/neoseurae12/post/0e578941-88f2-4dfc-a466-2ec0d3741f74/image.jpeg" width="500px">


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
<img src="https://velog.velcdn.com/images/neoseurae12/post/ae0b438d-d42d-422d-b4bf-3f993e1f6b26/image.jpeg" width="400px">


```python
A = np.array(
    [[0.644, 0.248, 0.108],
    [0.122, 0.701, 0.177],
    [0.055, 0.116, 0.829]])
A
```




    array([[0.644, 0.248, 0.108],
           [0.122, 0.701, 0.177],
           [0.055, 0.116, 0.829]])



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

    Rural ---> Metropolitan ---> Metropolitan ---> Metropolitan ---> Metropolitan ---> Metropolitan ---> Metropolitan ---> Metropolitan ---> Metropolitan ---> Rural ---> Rural ---> Metropolitan ---> stop


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
     [0.18825  0.339755 0.471996]


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
     [[0.18919621 0.33969614 0.47110765]
     [0.18919621 0.33969614 0.47110765]
     [0.18919621 0.33969614 0.47110765]] 
    
    π =  [0.18919621 0.33969614 0.47110765]


## Approach 3: Finding Left Eigen Vectors


```python
import scipy.linalg
values, left = scipy.linalg.eig(A, right=False, left=True)

print("left eigen vectors = \n", left, "\n")
print("eigen values = \n", values)
```

    left eigen vectors = 
     [[ 0.30972889  0.55521972 -0.34250589]
     [ 0.55610897 -0.79606747 -0.47063278]
     [ 0.7712398   0.24084775  0.81313867]] 
    
    eigen values = 
     [1.        +0.j 0.49293619+0.j 0.68106381+0.j]



```python
pi = left[:,0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
pi_normalized
```




    [0.18919620828463884, 0.3396961433580054, 0.4711076483573558]
