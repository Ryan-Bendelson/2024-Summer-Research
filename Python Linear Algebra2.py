#Install Scipy using pip command in Anaconda Prompt prior to running this file
import numpy as np #Import numpy as np
from scipy import linalg #Import the linear algebra module of Scipy
import math #Import the math library

#Create a function that returns the adjoint of a given matrix or vector
def getAdjoint(matrix):
    adjoint = matrix.transpose()
    adjoint = np.conjugate(adjoint)
    return adjoint

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
print("\nThis Bell system has a reduced density matrix for subsystem A of:\n", partialTrace2x2B(densityMatrixBell))
print("\nThis Bell system has a reduced density matrix for subsystem B of:\n", partialTrace2x2A(densityMatrixBell))

#Example 2 for partial traces
densityMatrixMixed = (1 / 2) * (np.kron(getAdjoint(state00), state00) + np.kron(getAdjoint(state11), state11))
print("\nA density matrix of a two-qubit system in a mixed state is:\n", densityMatrixMixed)
print("\nThis mixed state system has a reduced density matrix for subsystem A of:\n", partialTrace2x2B(densityMatrixMixed))
print("\nThis mixed state system has a reduced density matrix for subsystem B of:\n", partialTrace2x2A(densityMatrixMixed))

#Example 2 for partial traces
densityMatrixProduct = np.kron(getAdjoint(state01), state01)
print("\nA density matrix where the two qubits are independent is:\n", densityMatrixProduct)
print("\nThis system of independent qubits has a reduced density matrix for subsystem A of:\n", partialTrace2x2B(densityMatrixProduct))
print("\nThis system of independent qubits has a reduced density matrix for subsystem B of:\n", partialTrace2x2A(densityMatrixProduct))

#Create a function that will return a matrix of floats with value 0.0 of a given size
def getMatrix(numRows, numCols):
    matrix = np.array([0.0])
    for i in range(1, (numRows * numCols)):
        matrix = np.append(matrix, [0])
    matrix = matrix.reshape(numRows, numCols)
    return matrix

#Create a function that returns the partial trace over the last subsystem of a given Kronecker product
def partialTraceLast(kron):
    numRows = int(kron.shape[0] / 2)
    numCols = int(kron.shape[1] / 2)
    result = getMatrix(numRows, numCols)
    for i in range(0, numRows):
        for j in range(0, numCols):
            result[i, j] = kron[i, j] + kron[numRows + i, numCols + j]
    return result

#Create a function that returns the partial trace over subsystem A of a given Kronecker product
def partialTraceA(kron):
    numRows = int(kron.shape[0] / 2)
    numCols = int(kron.shape[1] / 2)
    result = getMatrix(numRows, numCols)
    i = 0
    j = 0
    while(i < (numRows * 2)):
        while(j < (numCols * 2)):
            result[int(i / 2), int(j / 2)] = kron[i, j] + kron[(i + 1), (j + 1)]
            j = j + 2
        j = 0
        i = i + 2
    return result

#Create a function that can pull out part of a matrix and return it as its own matrix
def getSubmatrix(matrix, startRow, startCol, endRow, endCol):
    numRows = endRow - startRow
    numCols = endCol - startCol
    submatrix = getMatrix(numRows, numCols)
    for i in range(startRow, endRow):
        for j in range(startCol, endCol):
            submatrix[(i - startRow), (j - startCol)] = matrix[i, j]
    return submatrix

#Create a function that will modify a given matrix by replacing a given part of it with a given matrix
def getModifiedMatrix(matModify, matInsert, startRow, startCol):
    for i in range(0, matInsert.shape[0]):
        for j in range(0, matInsert.shape[1]):
            matModify[(startRow + i), (startCol + j)] = matInsert[i, j]
    return matModify

#Create a function that will return the partial trace of a given Kronecker product over a given subsystem (A is 1)
def partialTrace(kron, qubit):
    numRows = int(kron.shape[0]) #Store the number of rows (same as number of columns) of the original Kronecker product density matrix
    result = getMatrix(int(numRows / 2), int(numRows / 2)) #Create a matrix to put the result of the partial trace in and return
    subdivisions = 4 ** (math.log2(numRows) - qubit) #Store how many submatrices will be taken from the original density matrix
    subrows = int(math.sqrt((numRows ** 2) / subdivisions)) #Store how many rows (same as number of columns) each submatrix will have
    submatrix = getMatrix(subrows, subrows) #Create a matrix to temporarily store each submatrix
    subtrace = getMatrix(int(subrows / 2), int(subrows / 2)) #Create a matrix to temporarily store each partial trace of a submatrix
    tempRow = 0 #Create a variable to store the current row index
    tempCol = 0 #Create a variable to store the current column index

    #Take submatrices of the original density matrix, take their partial traces, and put the partial traces into the final result matrix
    while((tempRow + subrows) <= numRows):
        while((tempCol + subrows) <= numRows):
            submatrix = getSubmatrix(kron, int(tempRow), int(tempCol), int(tempRow + subrows), int(tempCol + subrows))
            subtrace = partialTraceLast(submatrix)
            #print("subtrace:\n", subtrace)
            result = getModifiedMatrix(result, subtrace, int(tempRow / 2), int(tempCol / 2))
            tempCol = tempCol + subrows
        tempCol = 0
        tempRow = tempRow + subrows
    
    #print("\nresult:\n", result)
    return result

#Example 3 for partial traces
densityMatrix3Q = np.array([[0.125, 0, 0, 0, 0, 0, 0, 0.125], [0, 0.125, 0, 0, 0, 0, 0.125, 0], [0, 0, 0.125, 0, 0, 0.125, 0, 0], [0, 0, 0, 0.125, 0.125, 0, 0, 0], [0, 0, 0, 0.125, 0.125, 0, 0, 0], [0, 0, 0.125, 0, 0, 0.125, 0, 0], [0, 0.125, 0, 0, 0, 0, 0.125, 0], [0.125, 0, 0, 0, 0, 0, 0, 0.125]])
print("\nAn example of a three qubit matrix is:\n", densityMatrix3Q)
print("\nThe partial trace of this three qubit matrix taken over subsytem A is:\n", partialTrace(densityMatrix3Q, 1))
print("\nThe partial trace of this three qubit matrix taken over subsytem B is:\n", partialTrace(densityMatrix3Q, 2))
print("\nThe partial trace of this three qubit matrix taken over subsytem C is:\n", partialTrace(densityMatrix3Q, 3))

#Example 4 for partial traces
tester = np.array([[0.25, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.25, 0, 0, 0, 0, 0], [0, 0, 0, 0.25, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0.25]])
print("\nAnother example of a three qubit matrix is:\n", tester)
print("\nThe partial trace of this three qubit matrix taken over subsytem A is:\n", partialTrace(tester, 1))
print("\nThe partial trace of this three qubit matrix taken over subsytem B is:\n", partialTrace(tester, 2))
print("\nThe partial trace of this three qubit matrix taken over subsytem C is:\n", partialTrace(tester, 3))
