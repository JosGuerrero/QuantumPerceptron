import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Quantum computing 
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.circuit.library import CZGate
from sklearn.linear_model import LinearRegression
from qiskit.visualization import plot_histogram
from qiskit import transpile, assemble
from qiskit_aer import Aer
import qmltools as qml
import transforms as tr
from sklearn.metrics import accuracy_score
from testWeight import *
from qiskit_ibm_runtime import QiskitRuntimeService
import time
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


def getProjections(mfs, coh, th, qc, backend, nshots, qiskitService):
    # Her I need to make a huge Quantum Circuit 
    # Parameters
    rows, cols = np.shape(coh)
    rowsT, colsT = np.shape(th)
    projCoh = np.zeros(cols)
    projTh = np.zeros(cols)
    m = len(mfs) # mfs is the weight vector
    nq = 7

    mix = np.concatenate((coh, th), axis = 1)
    print(np.shape(mix))
    # Mod here:
    mixProj = qml.batchProj(mix, mfs, nq, nshots, backend, qiskitService) 
    
    return mixProj


def data(test, seedCoh, seedTh):

   #Parameters 
    numBits = 8

   

    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))


    # ********************* Data **********************


    # Making the random test set
    np.random.seed(seedCoh)
    indexCoh = np.random.randint(0, cols , test)
    np.random.seed(seedTh)
    indexTh = np.random.randint(0, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


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
    mixedStates = np.append(coh, th, axis = 1)
    
    return mixedStates
 

def getDifferences (single, multi, classic): 
    
    single = np.array(single)
    multi = np.array(multi)
    #classic = np.array(multi)
    
    diffSingle = classic - single

    diffMulti = classic - multi
    
    return diffSingle, diffMulti


def rmse(dif):
    mse = np.mean((dif)**2)
    return np.sqrt(mse) 

def main():
    
    rmseSingleList = []
    rmseMultiList = []
    cohSingList = []
    
    thSingList = []
    cohMultiList = []
    thMultiList = []

    # get a real backend from the runtime service

    service = QiskitRuntimeService(name='qmlserver')
    #backend = service.least_busy(operational = True, simulator = False, min_num_qubits = 7)
    backend = service.backend("ibm_aachen") #Computer with cz gates
    


    '''
    I need to check atleast 20 diferent data
    '''

    cohSeeds = [111,222,333,444,555,666,777,888,999,100,
                153,194,193,625,426,765,356,381,165,144]
    thSeeds = [120,239,348,458,567,912,923,945,856,523,
               567,457,489,152,642,434,613,234,456,164]
    #subset = int(input('subset: '))
    subset = 1
    for s in np.arange(len(cohSeeds))[:15]:

        # Get data and weight vector 
       
        mixedStates = data(subset, cohSeeds[s], thSeeds[s])
        path = '/home/dguerrerom/Documents/UNAM/PerceprontV1/tresholdTest/09245bestSingleCh.csv'
        w = pd.read_csv(path, delimiter = ',', header = None).to_numpy()
        
        # *The different excecutions begin 
        
        
        # Single subcircuit: list of N circuits , in future> n projections 
        
        qc_single = qml.getProjCircuits(7, mixedStates, w)
        singleProjs = []
        #singleProjs = [0,0]
        #'''
        for circuit in qc_single:
            singleProjs.append(tr.simulateAerBackend([circuit], 2**13, backend))

        # Mutliple subcircuits 
        qc_multi = qml.getLargeCircuit(mixedStates, w, 7, 2**13, backend, service)
        
        init = time.time()
        multiProjs_dict = tr.simulateMultiSCirc([qc_multi], 2**13, backend)
        final = time.time()
        
        print('Elapsed time for a single excecution :', final - init)
        #'''
        multiProjs = []
        #multiProjs = [0,0]
        #'''
       
        coupling = multiProjs_dict['11'] / 2**13
        print('Multi Projs: ')
        print(multiProjs_dict)
        multiResults = [multiProjs_dict['01'] / 2**13 + coupling, multiProjs_dict['10']/2**13 + coupling]
        #'''

        # Now the classical executions 
        classicProjs = []
        rows, cols = np.shape(mixedStates)
        
        for i in range(cols):
            psi = mixedStates[:,i]
            proj = np.dot(psi,w)
            cm = np.abs(proj)**2
            classicProjs.append(cm[0] / rows)
        

        # Get differences 
        


        single_dif, multi_dif = getDifferences(singleProjs, multiResults, classicProjs)
        
        cohSingle = single_dif[0]
        thSingle = single_dif[1]
        
        cohMulti = multi_dif[0]
        thMulti = multi_dif[1]
        
        # Get the differences for each case, coherent thermal
        cohSingList.append(cohSingle)
        thSingList.append(thSingle)

        cohMultiList.append(cohMulti)
        thMultiList.append(thMulti)

   #print('multiResults: ', multiResults)

    #np.savetxt('TwoQubitsappendedSubCircuitsResults.csv', counts, delimiter=',')
    
    #plotHist(cohCounts, thCounts, treshold, 'simulatedAppend', assisted_text)
    


    #rmseSingle = rmse(classicProjs, singleProjs)
    #rmseMulti = rmse(classicProjs, multiProjs)


    #rmseSingleList.append(rmseSingle)
    #rmseMultiList.append(rmseMulti)

    

    df = pd.DataFrame(
        {
            'Coh Single Circuit ': cohSingList, 
            'Coh Multi Circuit ': cohMultiList,
            'Th Single Circuit ': thSingList,
            'Th Mutliple Circuit ': thMultiList
            }
            )
    
    rmseDict = pd.DataFrame({
            'Coh Single Circuit ': 0, 
            'Coh Multi Circuit ': 0,
            'Th Single Circuit ': 0,
            'Th Mutliple Circuit ': 0
            }, 
            index = np.arange(4))

    for col in df:
        rmseDict[col] = rmse(df[col])
        df[col] = np.abs(np.array(df[col]))

    colors = {
            'Coh Single Circuit ':'red' , 
            'Coh Multi Circuit ': 'orange',
            'Th Single Circuit ': 'blue',
            'Th Mutliple Circuit ': 'purple'
            }
                
           
    

    df.to_csv('DifferencesBtwCircuits.csv')
    
    rmseDict.to_csv('RMSE_differencesCircu.csv')

    for col in df.columns:
        plt.scatter(df.index, df[col], label = col, color=colors[col])
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Differences')
    plt.title('Differences in modes of excecution')
    plt.savefig('DifferencesBtwCircuits.pdf', pad_inches='layout', bbox_inches = 'tight')

    #plt.show()
    
if __name__ == '__main__':
    main()
