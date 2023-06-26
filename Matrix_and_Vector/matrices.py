
import numpy as np

A = [[1,2,3], [4,5,6], [7,8,9]]
print ("A = ", A)
print("A[1] = ", A[1]) #2nd row
print("A[1][2] = ", A[1][2]) # thrid element of second row
print("A[0][-1]", A[0][-1]) # last element of 1st row

column = []; # empty list
for row in A:
    column.append(row[2])
print("3rd column = ", column)

B = [[11,12,13],[14,15,16], [17,18,19]]
result =[[0,0,0],
         [0,0,0],
         [0,0,0]]

# matrix addition using nested list comprehension
result = [[A[i][j] + B[i][j]  for j in range(len(A[0]))] for i in range(len(A))]

for r in result:
    print(r)
    
# NumPy is a package for scientific computing which has support for a powerful N-dimensional array object.
a = np.array([1,2,3])
X = np.array([[1,2,3],[4,5,6]])
print(X)
X= np.array([[1,2,3],[4,5,6]], dtype=float)
print(X)

# Array of zeros and ones
zeors_array = np.zeros((2,3))
print(zeors_array)
ones_array = np.ones((1,5))
print(ones_array)
C = np.array([[2,4], [1,3]])
D = np.array([[10,11],[100,200]])
E = C + D
print(E.ndim)
print(E.shape)
mul = [[0,0],[0,0]]

# program to multiply two matrices using nested loops
# iterate throught rows of C
for i in range(len(C)):
    #iterate through columns of Y
    for j in range(len(D[0])):
        #iterate through rows of Y
        for k in range(len(D)):
            mul[i][j]  += C[i][k] * D[k][j]
for r in mul:
    print(r)
    
print(C.dot(D))