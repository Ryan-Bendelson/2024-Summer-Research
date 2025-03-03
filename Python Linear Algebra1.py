#Install Scipy using pip command in Anaconda Prompt prior to running this file
import numpy as np #Import numpy as np
from scipy import linalg #Import the linear algebra module of Scipy
import math #Import the math library

#Working with a 3 by 3 identity matrix
I3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])#Store an identity matrix for a 3-dimensional vector space as I3
print("The 3 by 3 identity matrix is:\n", I3) #Display I3
I3_inv = linalg.inv(I3) #Store the inverse of matrix I3 as I3_inv
print("\nThe 3 by 3 identity matrix has an inverse of:\n", I3_inv) #Display I3_inv
print("\nThe determinant of the 3 by 3 identity matrix is", linalg.det(I3)) #Display the determinant of I3

#Using I3 on a vector
v1 = np.array([[1, 2, 3]]).transpose() #Create a 3-dimensional vector v1
print("\nThe vector v1 is:\n", v1) #Display vector v1
v1 = np.matmul(I3, v1) #Make v1 store the result of v1 multiplied by I3
print("\nAfter multiplying I3 with v1, the result is:\n", v1, "\nwhich is equivalent to v1.") #Display the new v1

#Multiplying vectors together
v2 = np.array([[4, 5, 6]]) #Create a 3-dimensional row vector v2
print("\nThe row vector v2 is:\n", v2) #Display row vector v2
A = np.matmul(v1, v2) #Store the product of v1 and v2 as A
print("\nMatrix A, the product of v1 and v2, is:\n", A) #Display A

#Multiplying matrices together
B = np.array([[1, 2, 3], [4, 5, 6]]) #Create a 2 by 3 matrix B
print("\nThe matrix B is:\n", B) #Display B
C = np.array([[10, 11], [20, 21], [30, 31]]) #Create a 3 by 2 matrix C
print("\nThe matrix C is:\n", C) #Display C
D = np.matmul(B, C) #Store the product of B and C as D
print("\nThe matrix D, which is the product of B and C, is:\n", D) #Display D

#Inner product of two vectors
v1_inner_v2 = np.inner(v1.transpose(), v2) #Store the inner product of v1 and v2 as v1_inner_v2
print("\nThe inner product of v1 and v2, is:\n", v1_inner_v2) #Display v1_inner_v2

#Outer product of two vectors
v1_outer_v2 = np.outer(v1, v2)#Store the outer product of v1 and v2 as v1_outer_v2
print("\nThe outer product of v1 and v2, is:\n", v1_outer_v2) #Display v1_outer_v2

#Tensor product of vectors
v1_tensor_v2 = np.kron(v1, v2.transpose()) #Store the tensor product of v1 and v2 as v1_tensor_v2
print("\nThe tensor product of v1 and v2 is:\n", v1_tensor_v2) #Display v1_tensor_v2

#Tensor product of matrices
B_tensor_C = np.kron(B, C) #Store the tensor product of B and C as B_tensor_C
print("\nThe tensor product of B and C is:\n", B_tensor_C) #Display B_tensor_C

#Pauli matrices in Python
pauliX = np.array([[0, 1], [1, 0]]) #Store the Pauli matrix X as pauliX
pauliY = np.array([[0, complex(0, -1)], [complex(0, 1), 0]]) #Store the Pauli matrix Y as pauliY
pauliZ = np.array([[1, 0], [0, -1]]) #Store the Pauli matrix Z as pauliZ

#Create a function that returns an identity matrix for a given matrix or column vector
def getIdentityMatrix(matrix):
    numRows = matrix.shape[0]
    I = np.array([0])
    for i in range(1, (numRows ** 2)):
        I = np.append(I, [0])
    I = I.reshape(numRows, numRows)
    for i in range(0, numRows):
        I[i, i] = 1
    return I

#Create a function that returns the adjoint of a given matrix or vector
def getAdjoint(matrix):
    adjoint = matrix.transpose()
    adjoint = np.conjugate(adjoint)
    return adjoint

#Create a function that returns the inner product of a given matrix and a given column vector which the matrix operates on
def getInnerMatrixProduct(matrix, vectorKet):
    vectorBra = getAdjoint(vectorKet)
    newKet = np.matmul(matrix, vectorKet)
    innerMatrixProduct = np.inner(vectorBra, newKet.transpose())
    return innerMatrixProduct

#Create a function that returns whether a list of vectors is orthonormal
def isOrthonormal(vectors):
    for i in vectors:
        if(math.sqrt(np.inner(i, i)) != 1):
            return False
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if(np.inner(vectors[i], vectors[j]) != 0):
                return False
    return True

#Create a function that returns an orthonormal basis for a vector space of a given dimension
def getOrthonormalBasis(dimensions):
    vector = np.array([1])
    for i in range(1, dimensions):
        vector = np.append(vector, [0])
    basis = np.array(vector)
    for i in range(1, dimensions):
        vector = [0]
        for j in range(1, dimensions):
            if(i == j):
                vector = np.append(vector, [1])
            else:
                vector = np.append(vector, [0])
        basis = np.append(basis, vector)
    basis = basis.reshape(dimensions, dimensions)
    return basis

#Create a function that returns the 2 by 2 partial trace matrix of a given 4 by 4 Kronecker product where each entry is the sum of corresponding sub-matrix entries
def partialTrace2x2A(kron):
    matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    matrix[0, 0] = kron[0, 0] + kron[2, 2]
    matrix[0, 1] = kron[0, 1] + kron[2, 3]
    matrix[1, 0] = kron[1, 0] + kron[3, 2]
    matrix[1, 1] = kron[1, 1] + kron[3, 3]
    return matrix

#Create a function that returns the 2 by 2 partial trace matrix of a given 4 by 4 Kronecker product where each entry is the trace of the corresponding sub-matrix
def partialTrace2x2B(kron):
    matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    matrix[0, 0] = kron[0, 0] + kron[1, 1]
    matrix[0, 1] = kron[0, 2] + kron[1, 3]
    matrix[1, 0] = kron[2, 0] + kron[3, 1]
    matrix[1, 1] = kron[2, 2] + kron[3, 3]
    return matrix

#Example 1 for partial traces
state0 = np.array([[1, 0]]).transpose()
state1 = np.array([[0, 1]]).transpose()
state00 = np.kron(state0, state0)
state01 = np.kron(state0, state1)
state10 = np.kron(state1, state0)
state11 = np.kron(state1, state1)
densityMatrixBell = (1 / 2) * (np.kron(getAdjoint(state00), state00) + np.kron(getAdjoint(state00), state11) + np.kron(getAdjoint(state11), state00) + np.kron(getAdjoint(state11), state11))
print("\nThe density matrix of a system with subsystems in the Bell state is:\n", densityMatrixBell)
print("\nThis Bell system has a reduced density matrix for subsystem A of:\n", partialTrace2x2A(densityMatrixBell))
print("\nThis Bell system has a reduced density matrix for subsystem B of:\n", partialTrace2x2B(densityMatrixBell))

#Example 2 for partial traces
densityMatrixMixed = (1 / 2) * (np.kron(getAdjoint(state00), state00) + np.kron(getAdjoint(state11), state11))
print("\nA density matrix of a two-qubit system in a mixed state is:\n", densityMatrixMixed)
print("\nThis mixed state system has a reduced density matrix for subsystem A of:\n", partialTrace2x2A(densityMatrixMixed))
print("\nThis mixed state system has a reduced density matrix for subsystem B of:\n", partialTrace2x2B(densityMatrixMixed))

#Example 2 for partial traces
densityMatrixProduct = np.kron(getAdjoint(state01), state01)
print("\nA density matrix where the two qubits are independent is:\n", densityMatrixProduct)
print("\nThis system of independent qubits has a reduced density matrix for subsystem A of:\n", partialTrace2x2A(densityMatrixProduct))
print("\nThis system of independent qubits has a reduced density matrix for subsystem B of:\n", partialTrace2x2B(densityMatrixProduct))
