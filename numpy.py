import subprocess as sp
tmp = sp.call('cls',shell=True)

import numpy as np

''' Conversion of Python list to Numpy array '''
#
#arr = np.array([2,5,6,8],dtype = int)
#print(arr)
#print(np.result_type(arr))
#arr = arr.astype(complex)
#print(np.result_type(arr))

''' Intrinsic Numpy array Creation '''

#print(np.zeros( (4,3) ))
## float64 is a by default dtype
#print(np.ones( (3,3), dtype = np.float64 ))
## arange() is a function similar to range()
## that returns arrays instead of lists
## It can accept float arguments too 
#print(np.arange(0, 10, 1.33, dtype = np.float64))
## arange gives uncertain number of values based on steps
## Hence, linspace, which asks for number of values
#print(np.linspace(0, 160, 5, dtype = np.float64))
## rand -> random values in a given shape
#print(np.random.rand(1,3))



''' Printing large numbers '''

## Ellipses is used to skip central part of an array
#print(np.arange(10000))



''' Matrix Creation '''

## Using reshape to convert array into matrix
#print(np.array([5,6,8,45,12,52]).reshape(2,3))
#
## Using matrix function
#print(np.matrix([[1,2],[3,4]]))
#print(np.eye(3)) # Identity matrix


''' Vectorized Operations '''

#arr1 = np.array([150, 200, 30])
#arr2 = np.arange( 3 )
#print(arr1 - arr2)
#
#print(arr2**3)
#
#print(arr2 < 1)
## Gives the index where new value can be placed to
## preserve order
#print(np.searchsorted(arr1, 100))
#print(np.searchsorted(arr2, 100))


## Matrix multiplication
#mat1 = np.array([[1, 2],
#                 [3, 4]])
#mat2 = np.array([[2, 3],
#                 [5, 6]])
#print(mat1 * mat2)  # Elementwise product
#print(mat1.dot(mat2))   # Dot product
#print(np.dot(mat1,mat2)) # Dot product


## Shorthand operations
#arr1 = np.ones((2,3), dtype = float)
#arr1 *= 5
#print(arr1)
#arr1 += 5
#print(arr1)


## Comparison Operations
#mat1 = np.matrix([[0, 20],
#                  [3, -6]])
#mat2 = np.matrix([[1, 12],
#                  [0.30, 4]])
#print(np.greater_equal(mat1, mat2))
#print(np.less(mat1, mat2))
#print(np.equal(mat1, mat2))
#print(np.not_equal(mat1, mat2))


## Logical Operations
#print(np.logical_and(1, False))
#print(np.logical_or(1, 0))
#print(np.logical_not(1))
#print(np.logical_xor(1, 0))


## Universal functions
#mat1 = np.matrix([[1, 2],
#                  [3, 4]])
#
#print(mat1.sum())
#print(mat1.min())
#print(mat1.max())
#print(mat1.shape)
## Using AXIS to specify the direction
## 0 -> Vertical
## 1 -> Horizontal 
#print(mat1.sum(axis = 0))
#print(mat1.sum(axis = 0).shape)
#print(mat1.sum(axis = 1))
#print(mat1.sum(axis = 1).shape)
#print(mat1.cumsum(axis = 0))
#print(mat1.cumsum(axis = 1))
#print(np.sqrt(mat1))
#print(np.sin(mat1))
#print(np.sin(np.pi/2))


## Matrix attributes
#mat1 = np.matrix([[1.2, 2-6j],
#                  [3+5j, 4]], dtype = complex)
#mat2 = np.matrix([[1, 2],
#                  [3, 4]], dtype = int)
#
#print(mat1.ndim)    # Number of dimensions
#print(mat1.size)    # Number of elements
#print(mat1.T)   # Transpose
#print(mat1.A1)  # 1D array
#print(mat1.H)   # Complex conjugate transpose
#print(mat1.imag)    # imag part
#print(mat1.real)    # real part
#print(abs(mat1))    # Absolute value
#print(mat1.itemsize) # total bytes occupied by 1 ele
#print(mat1.nbytes)  # total bytes consumed by matrix
#print(mat2.nbytes)


## Matrix methods
#mat1 = np.arange(9, dtype = np.float64).reshape(3,3)
#mat2 = np.array([[1, 2+3j],
#                 [3-9j,2]], dtype = complex)
#mat3 = np.array([[1, 4],
#                 [3, 2]])
#mat4 = np.array([[10, 40],
#                 [30, 20]])
#print(mat1)
#print(mat1.diagonal()) # Parameters -> 1, -1, 2, -2
#print(np.trace(mat1))   # Sum of diag. ele.
#
#print(mat1.compress([True,False,True],axis = 0)) #row
#print(mat1.compress([True,False,True],axis = 1)) #col
#
#print(mat2.conjugate())
#
## Indexing is alternate to itemset
#mat1.itemset(5,98)  # counts in row axis @ pos 5th
#mat1.itemset( (2,1), 45) # @ pos (2,1)
#print(mat1)
#
#mat3.sort(axis = 0) # Col wise
#print(mat3)
#mat4.sort(axis = 1) # Row wise
#print(mat4)



''' Indexing & Slicing '''

## 1D array
#arr1 = np.array([2, 5, 6, 4, 3])
#print(arr1[2])
#print(arr1[1:3])
#print(arr1[1:4:2])  # Every 2nd ele
#print(arr1[::2])    
#print(arr1[::-1])   # Reversed array
#print(arr1[::-2])   # Reversed array (every 2nd ele)


## Multidimensional array
#mat1 = np.arange( 9 ).reshape(3,3)
#print(mat1)
#print(mat1[2,2])
#print(mat1[:,2])
#print(mat1[1,:])
#print(mat1[0:2,2])  # doesn't consider last row
#print(type(mat1[1,:].shape))


## Advanced indexing
#
#arr1 = np.array([52, 65, 89, 78, 45, 11, 23, 96])
#arr1_ind = [1, 3, 5]
## Passing array as index
#print(arr1[arr1_ind])
#
#arr1_ind = [1, 3, 5]
## assigning values to index of array
#arr1[arr1_ind] = [-50, -60, -70]
#print(arr1)
#
## selection through indexing
#arr1_ind = arr1 > 50
#print(arr1)
#print(arr1[arr1_ind])
#
## repetitive index
#arr1_ind = [0,0,2,3]
#arr1[arr1_ind] = [100, 200, 300, 400] 
#print(arr1)
#
## Indexing on matrix
#mat1 = np.array([[ 0,  1,  2,  3],
#                 [ 4,  5,  6,  7],
#                 [ 8,  9, 10, 11]])
## size of indA and indB must be equal
#indA = np.array([[2,2],[1,1]])
#indB = np.array([[2,1],[1,0]])
#
#print(mat1[indA, indB])
#print(mat1[indA, 1])
#print(mat1[:, indB])
#mat1[indA, indB] = np.array([[100, 90],[50, 40]])
#print(mat1)



''' Manipulations '''

#mat1 = np.arange( 4 ).reshape(2,2)
#print(np.array(mat1.flat)) # Gives 1D array
#print(np.array(mat1.ravel()))  # Gives 1D array
## With -1 other dim are  calc. automatically
#print(mat1.reshape(4,-1).shape)
#print(mat1.reshape(-1,4).shape) 
#print(mat1)     # No perm. change to mat1 yet
#mat1.resize( (1,4) )    # Perm. change
#print(mat1)


## Stacking arrays to form matrix
#
#arr1 = np.array([[1, 2],
#                 [3, 4]])
#arr2 = np.array([[10, 20],
#                 [30, 40]])
## VSTACK -> adds row(s)
## even matrix and 1D arrays can be stacked
#matR = np.vstack( (arr1, arr2) ) 
#print(matR)
#print(matR.shape)
#
## HSTACK -> adds col(s)
#matC = np.hstack( (arr1, arr2) ) 
#print(matC)
#print(matC.shape)


## Splitting of arrays 
#
#arr1 = np.arange(16).reshape(4,4)
#
## VSPLIT -> Split row wise
#matR = np.vsplit(arr1,4)
#print(np.array(matR))
#print(np.array(matR).shape)
#print(matR[0][0][2]) # Currently, a normal list
#
## HSPLIT -> Split col wise
#matR = np.hsplit(arr1,4)
#print(np.array(matR))
#print(np.array(matR).shape)
#print(matR[0][1][0]) # Currently, a normal list


## Copying the value
#arr1 = np.arange(12).reshape(4,3)
#arr2 = arr1
#print(arr2 is arr1)
#arr2[2,2] = 50 
## mutability plays its role
#print(arr1)
#
## Deep copy
#arr1 = np.arange(12).reshape(4,3)
#arr2 = arr1.copy()
#print(arr2 is arr1)
#arr2[2,2] = 50 
#print(arr1)
#print(arr2)



''' Linear Algebra '''

#mat1 = np.arange(4).reshape(2,2)
#mat2 = (np.arange(4)*2).reshape(2,2)
#mat3 = (np.arange(4)*3).reshape(2,2)
#
## Performing multiple dot product in one go
#print(np.linalg.multi_dot( [mat1, mat2, mat3] ))
#
## Finding Inverse of matrix
#print(np.linalg.inv(mat2))
#
## Computing real eigenvalues and eigenvectors
#e1, e2 = np.linalg.eig(np.diag( (1, 2, 3) ))
#print(e2)
#print(e1)
#
## Alternate method to get evalues
#print(np.linalg.eigvals(np.diag( (1, 2, 3) )))
#
## Computing determinant of matrix
#print(np.linalg.det(mat1))
#
## Computing matrix rank
#print(np.linalg.matrix_rank(mat1))
#
## Solving system of linear equations
## 3x + y = 9
## x + 2y = 8
#
#a = np.array([[3, 1],[1, 2]])
#b = np.array([9, 8])
#print(np.linalg.solve(a, b))
