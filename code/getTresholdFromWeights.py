import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import bitarray as bta
import bitarray.util as butil
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import normalize
import time

#Quantum computing 
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.circuit.library import CZGate
from qiskit.visualization import plot_histogram
from qiskit import transpile, assemble
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService

from testWeight import *
import qmltools as qml
import transforms as tr


# Do the projections 

def scale(num, resolution, max_value):
    return np.floor(resolution * num / max_value)

def getBars(meanPhotonNumber, datapoints, path):
    name = 'H'+str(meanPhotonNumber)+'_M_'+str(datapoints)+'.mat'
    return sio.loadmat(path+name)

def bars2bit(database, numBits):
    rows, cols = np.shape(database)

    newDataBase = np.zeros((rows*numBits,cols), dtype = np.int8) #Storage of new binvect
    #print(np.shape(newDataBase))
    resolution = 2**numBits

    newRow = np.zeros(numBits, dtype = np.int8)
    maxVal = 1

    init = time.time()
    for i in range(cols):
        #Initial binary string
        newRow = bta.bitarray('')

        for j in range(rows):

            num = np.int64(scale(database[j][i], resolution, maxVal)) # Scale the number
            binaryNum = np.binary_repr(num, numBits) # Convert it into a binary string

            newRow.extend(binaryNum) # Append the entire string into de bitarray newRow


        binaryVec = np.array(newRow.tolist(), dtype = np.int8) # convert the bitarray into a np array of numBits*numBars lenght

        newDataBase[:,i] = binaryVec # assign the new vector to the column of the database
    final = time.time()

    print(final - init)
    return newDataBase

def bin2ones(database):
    rows, cols = np.shape(database)
    onesDatabase = -np.ones((rows, cols), dtype = np.int8)
    return onesDatabase**database



def getData(meanPhotonNumber, datapoints, numBits):

    path = '/home/dguerrerom/Documents/UNAM/Datos_Histogramas/'
    data = getBars(meanPhotonNumber, datapoints, path)
    cohData = data['Coh']
    thData = data['Th']
    #print(np.shape(cohData))

    enCohData = bars2bit(cohData, numBits)
    enThData = bars2bit(thData, numBits)

    return enCohData, enThData


def getProjections(mfs, coh, th, qc, backend, nshots):

    # Parameters
    rows, cols = np.shape(coh)
    rowsT, colsT = np.shape(th)
    projCoh = np.zeros(cols)
    projTh = np.zeros(cols)
    m = len(mfs) # mfs is the weight vector
    nq = 7

    for i in range(cols):
        #init = time.time()
        psi_i = coh[:,i]
        #projCoh[i] = qml.singleProjAer(psi_i, mfs, nq, nshots)
        projCoh[i] = qml.singleProjFake(psi_i, mfs, nq, nshots, backend)
        #cm = mfs@psi_i
        #projCoh[i] = np.abs(cm/m)**2

        psi_i = th[:,i]
        projTh[i] = qml.singleProjFake(psi_i, mfs, nq, nshots, backend)
        #projTh[i] = qml.singleProjAer(psi_i, mfs, nq, nshots)
        #cm = mfs@psi_i
        #end = time.time()
        #projTh[i] = np.abs(cm/m)**2
        #print('single proj time: ', end - init)
        #'''

    mix = np.concatenate((projTh, projCoh))
    histRange = (np.min(mix), np.max(mix))
    bins_n = 50
    cohHist = np.histogram(a = projCoh, bins = 'auto', range = histRange)
    n_bins = len(cohHist[1])
    thHist = np.histogram(a = projTh, bins = n_bins , range = histRange)
    
    return (projCoh, projTh)
 
    

def main():

   #Parameters 
    numBits = 8

    # Data

    subs = int(input('Subs >= 5:'))
    if subs < 2 : raise Exception("Subs  must be grater or equal to 5")
    coh, th = getData(77, 160,numBits)

    ## Subset
    coh, th = bin2ones(coh), bin2ones(th)
    coh, th = coh[:, :subs], th[:, :subs]

    cohRows, cohCols = np.shape(coh)
    thRows, thCols = np.shape(th)

    # Mixing 
    mixedStates = np.append(coh, th, axis = 1)
    mixRows, mixCols = np.shape(mixedStates)

    '''
    Quantum Circuit 
    '''

    N = 7
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(2, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    nshots = 100

    # Backend for fake simulation

    # The backend needs to be outside the excecution loop
    # get a real backend from the runtime service

    service = QiskitRuntimeService()
    backend = service.least_busy(operational = True, simulator = False, min_num_qubits = qc.num_qubits)
    backend = service.backend("ibm_brisbane")

    # Filler section:  We have a vector of dim 56 in a dim 64 H space
    fillerAllPos = np.ones(2**(qc.num_qubits - 1) - mixRows, dtype = np.int8) # Remaining slots are filled with ones 
    # [1 -1 1 -1 -1] [1 1 1 1]
    fillerHalfNeg = fillerAllPos

    #Adding negative ones 
    fillerHalfNeg[int(len(fillerAllPos)/2):] = -1




    # Filler in data 
    fillerMatrix = np.ones((2**(qc.num_qubits - 1) - mixRows, cohCols), dtype = np.int8)
    coh = np.concatenate((coh, fillerMatrix), axis = 0)
    th = np.concatenate((th, fillerMatrix), axis = 0)

    # Data projections
    path = str(input('weight(filename): '))
    hint = int(input('Assisted (0), Non-Assisted (1) : '))
    w = pd.read_csv(path, delimiter = ',', header = None).to_numpy()
    

    cohProjs, thProjs = getProjections(w, coh, th, qc, backend, 2**8)
    

    # Treshold 
   
    textAssisted = 'assisted' *int(not(hint))
    treshold = fakeTypeTest(hint, cohProjs, thProjs,textAssisted)
    
    # Test 

    testClass, testQiskit, testFake = globalTest(w, subs,treshold,textAssisted, backend)





if __name__ == '__main__' : 
    main()





