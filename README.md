```python
import numpy as np
```

# The Dynamics of Internal Migration in S. Korea: A Markov Chain Analysis


<img src="https://velog.velcdn.com/images/neoseurae12/post/c5bd58b7-0033-4850-a198-000ea5bba435/image.jpeg" width="500px">


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
<img src="https://velog.velcdn.com/images/neoseurae12/post/c2b43497-b224-45bf-97a8-bb8063be890e/image.jpeg" width="400px">
<img src="https://velog.velcdn.com/images/neoseurae12/post/e45935ed-7cb5-4509-a30e-69f7edd48f30/image.jpeg" width="400px">


```python
A = np.array(
    [[0.646, 0.246, 0.108],
    [0.121, 0.703, 0.176],
    [0.055, 0.115, 0.830]])
A
```




    array([[0.646, 0.246, 0.108],
          [0.121, 0.763, 0.116],
          [0.055, 0.076, 0.869]])



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

    Rural ---> Rural ---> Rural ---> Rural ---> Rural ---> Rural ---> Rural ---> Seoul ---> Metropolitan ---> Metropolitan ---> Seoul ---> Seoul ---> stop


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
     [0.189963 0.346825 0.463213]


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
     [[0.19033879 0.3461904  0.46347081]
      [0.19033879 0.3461904  0.46347081]
      [0.19033879 0.3461904  0.46347081]] 

    π =  [0.19033879 0.3461904  0.46347081]


## Approach 3: Finding Left Eigen Vectors


```python
import scipy.linalg
values, left = scipy.linalg.eig(A, right=False, left=True)

print("left eigen vectors = \n", left, "\n")
print("eigen values = \n", values)
```

    left eigen vectors = 
     [[ 0.31254282  0.69473091 -0.22720446]
     [ 0.5684565  -0.71886475 -0.56557639]
     [ 0.76103495  0.02413384  0.79278085]] 
    
    eigen values = 
     [1.        +0.j 0.52270727+0.j 0.75529273+0.j]



```python
pi = left[:,0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
pi_normalized
```




    [0.19033879294844838, 0.3461903987259954, 0.46347080832555626]
