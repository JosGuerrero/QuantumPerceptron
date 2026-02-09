import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 
import bitarray as bta 
import bitarray.util as butil 
import transforms as tr 
import time 
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService


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
from qiskit.circuit.library import MCXGate, UnitaryGate
from qiskit.circuit.library import ZGate, XGate
from qiskit.circuit.library import MCMT
from HouseHolder import getUnitaryHH, getUiSimple
#from plotAll import quantumRealProd, quantumProd
import transforms as tr 


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

def overlap(coherent, thermal):
    np.shape(coherent)
    np.shape(thermal)
    return np.sum(np.sqrt(coherent) * np.sqrt(thermal))**2 / ((np.sum(coherent))*(np.sum(thermal)))


def normalizeRange(cohHist, thHist):
    leftTh = thHist[1][0]
    rightTh = thHist[1][len(thHist[1]) - 1]
    leftCoh = cohHist[1][0]
    rightCoh = cohHist[1][len(cohHist[1]) - 1]
    print(rightCoh - rightTh)
    fillerTh = np.zeros(int(np.sqrt(rightCoh**2 - rightTh**2)))
    fillerCoh = np.zeros(int(np.sqrt(leftTh**2 - leftCoh**2)))

    newCoh = np.concatenate((fillerCoh,cohHist[0]))
    newTh = np.concatenate((thHist, fillerTh[0]))
    return newCoh, newTh


def getTresholdInLoop(cohHist, thHist):

    mix = np.concatenate((cohHist[1], thHist[1]))
    min = np.min(mix)
    max = np.max(mix)
    mixArange = np.arange(min, max)
    treshold = np.median(mixArange)
    return treshold

def multipointCO(childrenMat):

    l, col = np.shape(childrenMat)
    #print(l,col)
    probArr = np.ones(l)/ l
    #print(probArr)
    ran = np.arange(0, col, 2)

    for i in ran:
        chosen = np.random.choice(np.arange(l).tolist(), 1, probArr.tolist())  
        tmp1 = childrenMat[chosen, i]
        tmp2 = childrenMat[chosen, i + 1]

        childrenMat[chosen, i] = tmp2
        childrenMat[chosen, i + 1] = tmp1

    return childrenMat

def randomMutation(population, mutVect):
    #Mut vect are the selected columns 
    cols = np.nonzero(mutVect)
    randomId = np.random.randint(0, len(population), len(cols)) # randomId are the selected entries, one per selected chromosome
    population[randomId, cols] = -1*population[randomId, cols] #switch selected mutants 

    return population





def quantumProd(qc, mfs, psi_i, nshots):
        
    ''' Quantum Circuit: Preparation '''
    qc.clear()
    N = qc.num_qubits
    psi_plus = np.ones(2**(qc.num_qubits - 1))

    # Hadammard
    qc.h(np.arange(N - 1).tolist())
    
    # Creamos las transformaciones unitarias 
    
    # Ui
    #uiMat = getUnitaryHH(psi_plus, psi_i)
    # Testing the new simple method 
    uiMat = getUiSimple(psi_i)
    #print(uiMat)
    # Ancilla
    uiMatAnc = uiMat
    #uiMatAnc = np.kron(np.eye(2), uiMat)
    
    # Add it to the circuit
    ui = UnitaryGate(uiMatAnc)
    target = np.arange(N - 1).tolist()
    qc.append(ui, target)
    #print('StVec for psi_i=', psi_i)
    print('Numerical Vector:', psi_i)
    #print('State Vector:', Statevector(qc))
    st = Statevector(qc)
    # Check transforms 
    checkUI(st, psi_i, 64)
    
    
    # Uw 
    
    #uwMat = getUnitaryHH(mfs, psi_plus)
    # Testing the new get Uw method
    uwMat = getUiSimple(mfs)
    # Ancilla
    #uwMatAnc = np.kron(np.eye(2), uwMat)
    uwMatAnc = uwMat
    # Add it to the circuit
    uw = UnitaryGate(uwMatAnc)
    
    
    qc.append(uw, target)

    
    # Contract the state to | 0 > tensor N 
    #qc.h(np.arange(N - 1).tolist())
    # Now flip it to | 1 > tensor N 
    #qc.x(np.arange(N - 1).tolist())

    # Entangle with the ancilla
    qc.mcx(np.arange(N - 1).tolist(), N - 1)
    
    qc.measure(N - 1,0)
    print(qc)

    '''Simulating the circuit'''
    q_prod_test = np.sqrt(tr.simulate([qc], nshots)['1'] / nshots)

    return q_prod_test

def addFiller(data, num_qubits):
    # Filler section: The residual slots need to be filled 
    rows, cols = np.shape(data)
    
    fillerAllPos = np.ones(2**(num_qubits - 1) - rows, dtype = np.int8) # Remaining slots are filled with ones 
    # [1 -1 1 -1 -1] [1 1 1 1]
    fillerHalfNeg = fillerAllPos 

    #Adding negatives
    fillerHalfNeg[int(len(fillerAllPos)/2):] = -1 

    # Filler in data 
    fillerMatrix = np.ones((2**(num_qubits - 1) - rows, cols), dtype = np.int8)
    data = np.concatenate((data, fillerMatrix), axis = 0)
    
    return data

def getRSV(qc):
    nqubits = qc.num_qubits - 1
    
    sv_full = Statevector(qc).data.real
    #print(sv_full)
    sv = sv_full[:2**(qc.num_qubits - 1)]
   
    
    
    sv_real = np.sign(sv).astype(int) # cancels normalization
   
    return sv_real

def getQubitsId(decimalIndices, nqubits):
    qtargCtrl = []
    for i in decimalIndices:
        qtargCtrl.append(
            butil.int2ba(int(i), nqubits).tolist()
            )
    # convert it to np array
    #print('Error indices in binary: \n', qtargCtrl)
    qtargCtrl = np.array(qtargCtrl)

    # Now get the non zero indices 

    nonZeroId = []
    
    #Each element will have an array with the indexes where the basis are not zero 
    # The numeration of binary numbers are done from right to left, so I need to flip the array
    
    for base in qtargCtrl:
        nonZeroId.append(
                np.nonzero(np.flip(base))
                )
    # Now apply the control z gates using nonZeroId, the last as target and the other elements as control qubits
    
    #print('Qubits to work with: ')
    #print(nonZeroId)
    return nonZeroId

def mCZ(qc, clbits, target_ctr):
    z_gate = ZGate()
    multi_controlled_z = z_gate.control(clbits)
    qc.append(multi_controlled_z, target_ctr)
def mCX(qc, clbits, target_ctr):
    x_gate = XGate()
    multi_controlled_x = x_gate.control(clbits)
    qc.append(multi_controlled_x, target_ctr)
def flipBaseZero(qc, data):
        print('zero base encounter')

        z_gate = ZGate()
        clbits = qc.num_qubits - 2
        multi_controlled_z = z_gate.control(clbits)
        target_ctr = np.arange(clbits + 1).tolist()
        qc.append(multi_controlled_z, target_ctr)
        print(getRSV(qc))
        print('targ: ', target_ctr)
        qc.x(target_ctr)
        print(qc)
        sv_real = getRSV(qc)
        print('New state vector:')
        print(sv_real)
        
        oneBitRepr = []
        n_qubits = qc.num_qubits - 1 #Minus the anciliar   
        isin_basis = np.isin(data, sv_real, invert = True)
        errors = np.nonzero(
                    isin_basis #oneBitRepr saves the order, it lets me use it's index as qubit number
                )[0]
        print('Errors(' + str(len(errors))+'):')
        print(errors)


def applyCZ(qc, decimalIndices, nqubits):
    # The case of controlled z is more complicated
    # Once I get the bases where I want my amplitude shifts
    # from these bases I need to know wich qubits are in ones
    # then use all except the last one as control qubits and
    # the last one as target. 

    # Find the binary representations of the bases 
    decimalIndices = np.array(decimalIndices)
    
    nonZeroId = getQubitsId(decimalIndices, nqubits)
    z_gate = ZGate()
    for index in nonZeroId:
        
        index = np.array(index[0])
        print('index:', index)
        if(index.size > 1):
            control = index[:-1].tolist()
            target = index[-1].tolist()
            clbits = len(control)
            print('clbits: ', clbits)
            print('target: ', target, 'control: ', control)
            old_rsv = getRSV(qc).tolist()
            # Add multi-controlled Z 
            multi_controlled_z = z_gate.control(clbits)
            qc.append(multi_controlled_z, index.tolist())
            #mcmt_z = MCMT(ZGate(), num_ctrl_qubits=clbits, num_target_qubits=1)
            #qc.append(mcmt_z, index.tolist())  # Controls first, then target    
            #qc.cz(control_qubit = control, target_qubit = target)
            rsv = getRSV(qc)
            print(rsv)
            #changed_basis = np.nonzero(rsv.tolist() == old_rsv)
            #print('changed basis: ', changed_basis)
            

            #qc.mcp(-1,control, target)
            #Multi controlled z gates are not available, so I'll simulate them with cz gates 
        '''
        elif(index.size == 0):
            clbits = qc.num_qubits - 2
            multi_controlled_z = z_gate.control(clbits)
            target_ctr = np.arange(clbits + 1).tolist()
            qc.append(multi_controlled_z, target_ctr)
            qc.x(target_ctr)
            print('new: \n', getRSV(qc))
        #'''
    return qc 

def singleQubitStep(qc, sv_real, data):
    print('State goal: ', data)
    print('State recieved: ', sv_real)
    # Goal> Apply z gates in the basis where only one qubit needs to be flipped
    print(str('*'*5)+  'Single Qubit Step' + str('*'*5))    
    # Defining the target
    oneBitRepr = []
    n_qubits = qc.num_qubits - 1 #Minus the anciliar   
    isin_basis = np.isin(data, sv_real, invert = True)
    errors = np.where(sv_real != data)[0]
    print('Errors(' + str(len(errors))+'):')
    print(errors)

    # Now from these errors, wich can be represented with only one qubit?
    
    if errors[0] == 0:
      flipBaseZero(qc, data) 
    
    for i in np.arange(0, n_qubits):
        oneBitRepr.append(2**i) # The numbers that can be represented with only one bit, values from 1 to 32
        
    loneQubits = np.nonzero(
                np.isin(errors, oneBitRepr) #oneBitRepr saves the order, it lets me use it's index as qubit number 
            )[0]
    # Now I have an array with the index(bits) of lone quibts that needs a minus
    print('Errors that can be represented with only one qubit:')
    loneQubitsId = errors[loneQubits]
    print(loneQubitsId)
    
    # These basis are represented with just one qubit so 
    # to get the targets we can just take their log base 2 value
    
    target = np.log2(loneQubitsId).astype(int)
    print('Qubits to be flipped: \n', target)

    # Apply a z to those qubits
    for index in target:
        print(index)
        qc.z(index)
        print(getRSV(qc))
    return qc

def multiQubitStep(qc, sv_real_multi, data):
    # Goal> Apply cz gates in the basis where more than one qubit needs to be flipped
    print(str('*'*8)+  'Multi Qubit Step' + str('*'*8))
    print('SV goal: ', data)
    print('SV recieved: ', sv_real_multi)
    # Defining the target
    oneBitRepr = []
    n_qubits = qc.num_qubits - 1 #Minus the anciliar 
    
    #isin_basis_multi = np.isin(sv_real_multi, data, invert = True)
    errorsMulti = np.where(sv_real_multi != data)[0]
    #isin_basis = sv_real != data
    # Get where the vector state and the data state do not concide 
    #print(isin_basis)
    #print(isin_basis_multi)
    
    print('Errors(' + str(len(errorsMulti))+'): \n', errorsMulti)
    
    # Now from these errors, wich can be represented with more than one qubit?
    #if errorsMulti[0]==0: 
    #    flipBaseZero(qc, data)
    for i in np.arange(0, n_qubits - 1):
        oneBitRepr.append(2**i) # The numbers that can be represented with only one bit, values from 1 to 32
    


    # Now we take the indices of the elements not present in the one bit representation array 
    multiQubits = np.nonzero(
                np.isin(errorsMulti,oneBitRepr, invert = True) #oneBitRepr saves the order, it lets me use it's index as qubit number 
            )[0]
    
    # Now I have an array with the index(bits) of multi quibts that needs a minus
    print('Errors that needs  more than one qubit (must be all):')
    
    indices = errorsMulti[multiQubits]
    print(indices)
    print('Are they equal? '+ str(np.array_equal(indices ,errorsMulti)))
    # Apply the control Z gates

    qc = applyCZ(qc,indices,n_qubits)
    print('mqs:')
    print(qc)
    
    return qc

def removeFlips(qc,sv_real, data):
    print(str('*'*5)+  'Remove Non Desired Phases' + str('*'*5))
    print('SV goal: ', data)
    print('SV recieved: ', sv_real)

    # Goal> Apply cz gates in the basis where are unwanted flips
    
    n_qubits = qc.num_qubits - 1 #Minus the anciliar 
    isin = sv_real != data
    errors = np.nonzero(
                isin #oneBitRepr saves the order, it lets me use it's index as qubit number 
            )[0]
    # Now I have an array with the index(bits) of bases where there are unwanted flips
    print('Errors(' + str(len(errors))+'): \n',errors)
    # Apply a cz to those basis
    
    if np.size(errors) != 0:
        qc = applyCZ(qc,errors,n_qubits)


    return qc

def uiHG(qc, data):
    print(str('*'*8)+  'uiHG()' + str('*'*8))
    sv = getRSV(qc)
    qc = singleQubitStep(qc, sv, data)
    sv = getRSV(qc)
    qc = multiQubitStep(qc, sv, data)
    print(qc)
    sv =getRSV(qc)
    qc = removeFlips(qc, sv, data)
    
    return qc 

def uwHG(qc, w):
    '''
    Quantum Circuit 
    '''
    print('************************Performing Uw*******************************')
    N = qc.num_qubits 
    Nc = 1 # I need one bit per ancilliar qubit
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla') #And an ancilliar qubit per each data
    cr = ClassicalRegister(Nc, 'c')

    qcAux = QuantumCircuit(qr,ar,cr)
    qcAux.h(np.arange(N-1).tolist())
    qcAux = uiHG(qcAux, w)

    # Now replicate the circuit 
    qc.compose(qcAux)

    return qc

def testUiwHG():
    N = 7
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(2, 'c')

    qcUi = QuantumCircuit(qr,ar,cr)
    qcUw = QuantumCircuit(qr,ar,cr)
    nshots = 2**13
    cols = 1
    m = 64
    target = np.arange(N-1).tolist()
    data = (-1)**(np.random.randint(0,2,m))
    w = (-1)**(np.random.randint(0,2,m))
    #data[0] = -1
    #data = (-1)**np.array(
    #butil.int2ba(45693933234523451, 64).tolist())
        
    # Ui test 

    print('i vect: ', data)
    qcUi.h(target)
    print('Initial SV: ', getRSV(qcUi))
    uiHG(qcUi, data)
    finalSV = getRSV(qcUi)
    print('Data(goal): ', data)
    print('Finial SV:', finalSV)
    print('Are they equal? ', np.isin(data, finalSV))
    
    # Uw test 
    
    print('*************************************************************************************************************************************************************************************')
    #Prepare Psi_w
    qcUw.h(target)
    qcUw = uiHG(qcUw, w)

    print('Init in w? ',getRSV(qcUw) == w)
    
    #Now collapse it into |1...1>
    qcUw = uwHG(qcUw, w)
    print(getRSV(qcUw))
    
def getProjCircuitsHG(num_qubits, data, w):
    
    rows, cols = np.shape(data)

    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    #nshots = 100

    circuits = []
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    for i in range(cols):
        qc.clear()

        # Initialize in | + >
        qc.h(target)

        # Hypergraphs algorithms
        qc = uiHG(qc, data[:,i])
        qc = uwHG(qc, w)

        # Entangle the qubits with the ancilla qubit 
        qc.mcx(target, N - 1)
        # Measure the ancilla qubit 
        qc.measure(N - 1, 0)
        print(qc)
        circuits.append(qc)
    return circuits
    # Now we create the 


def getProjCircuits(num_qubits, data, w):
    
    rows, cols = np.shape(data)

    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    #nshots = 100

    circuits = []
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    for i in range(cols):
        qc.clear()

        # Initialize in | + >
        qc.h(target)

        # Unitary transforms 
        uiMat = getUiSimple(data[:,i])
        uwMat = getUiSimple(w) 
        print('Gate shape', np.shape(uiMat))
        #print(uiMat)
        #print(uwMat)
        np.savetxt('ui.txt', uiMat)
        # Transform them into unitary gates 
        ui = UnitaryGate(uiMat)
        uw = UnitaryGate(uwMat)

        # Add them to the circuit (build the perceptron) 
        qc.append(ui, target)
        qc.append(uw, target)
        qc.h(target)
        qc.x(target)

        # Entangle the qubits with the ancilla qubit 
        qc.mcx(target, N - 1)
        # Measure the ancilla qubit 
        qc.measure(N - 1, 0)
        print(qc)
        circuits.append(qc)
    return circuits
    # Now we create the 

def singleProjAer(i,w, num_qubits, n_shots):
    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    # Initialize in | + >
    qc.h(target)

    # Unitary transforms 
    uiMat = getUiSimple(i)
    uwMat = getUiSimple(w) 
    #print('Gate shape', np.shape(uiMat))
    #print(uiMat)
    #print(uwMat)
    np.savetxt('ui.txt', uiMat)
    # Transform them into unitary gates 
    ui = UnitaryGate(uiMat)
    uw = UnitaryGate(uwMat)

    # Add them to the circuit (build the perceptron) 
    qc.append(ui, target)
    qc.append(uw, target)
    qc.h(target)
    qc.x(target)

    # Entangle the qubits with the ancilla qubit 
    qc.mcx(target, N - 1)
    # Measure the ancilla qubit 
    qc.measure(N - 1, 0)
    print(qc)
    

    simJob = tr.simulate_sv([qc], n_shots)
    simResult = tr.getResultsFromJobs(simJob)
    return simResult



def singleProjFake(i,w,num_qubits,n_shots, backend):
    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    # Initialize in | + >
    qc.h(target)
    #stateVec = Statevector(qc)
    #print(stateVec)


    # Unitary transforms 
    uiMat = getUiSimple(i)
    uwMat = getUiSimple(w) 
    #print('Gate shape', np.shape(uiMat))
    #print(uiMat)
    #print(uwMat)
    np.savetxt('ui.txt', uiMat)
    # Transform them into unitary gates 
    ui = UnitaryGate(uiMat)
    uw = UnitaryGate(uwMat)

    # Add them to the circuit (build the perceptron) 
    qc.append(ui, target)
    qc.append(uw, target)
    qc.h(target)
    qc.x(target)

    # Entangle the qubits with the ancilla qubit 
    qc.mcx(target, N - 1)
    # Measure the ancilla qubit 
    qc.measure(N - 1, 0)
    #print(qc)
    

    simResult = tr.simulateAerBackend([qc], n_shots, backend)
    
    return simResult


def singleProjReal(i,w,num_qubits, n_shots):
    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    # Initialize in | + >
    qc.h(target)

    # Unitary transforms 
    uiMat = getUiSimple(i)
    uwMat = getUiSimple(w) 
    print('Gate shape', np.shape(uiMat))
    #print(uiMat)
    #print(uwMat)
    np.savetxt('ui.txt', uiMat)
    # Transform them into unitary gates 
    ui = UnitaryGate(uiMat)
    uw = UnitaryGate(uwMat)

    # Add them to the circuit (build the perceptron) 
    qc.append(ui, target)
    qc.append(uw, target)
    qc.h(target)
    qc.x(target)

    # Entangle the qubits with the ancilla qubit 
    qc.mcx(target, N - 1)
    # Measure the ancilla qubit 
    qc.measure(N - 1, 0)
    print(qc)
    

    simJob = tr.realExecute(qc)
    simResult = tr.getResultsFromJobs(simJob)
    return simResult

def getResultsBatch(job_id, qservice):
    counts_batch = []
    # Altough the job stands for all circuits, te counts are separated

    job = qservice.job(job_id)
    job_result = job.result()
    
    # Get counts for all particular oub results
    nshots = job_result[0].data.c.num_shots

    # Get couts 
    for pub in job_result:
        counts = pub.data.c.get_counts()['1']
        counts_batch.append(counts/nshots)# c stands for classical register
        
    return counts_batch


def getGatesDistro(isa_circuits, backend):
    isa_circuit = isa_circuits[0]
    gates = len(isa_circuit.data)
    qubits = isa_circuit.num_qubits
    circuitGates = []
    for i in range(gates):
        circuitGates.append(isa_circuit.data[i][1][0]._index)

    values, counts = np.unique(ar = np.array(circuitGates), return_counts = True)



    plt.bar(x = values, height = counts)
    plt.title(backend.name + ' :' + str(gates) + ' gates')
    plt.xlabel('qubit')
    plt.ylabel('Gates')
    plt.savefig('gatesDistro_'+str(backend)+'.pdf')
    plt.show()
    plt.close()
    return values, counts


def batchProj(data,w,num_qubits,n_shots, backend, qiskitService):
    circuits = getProjCircuits(num_qubits, data, w)
    simResults_job_id = tr.sendBatch(circuits, n_shots, backend)
    results = getResultsBatch(simResults_job_id, qiskitService)
    np.savetxt('batchRes.csv', results, delimiter=',')
    return results


def batchProjHG(data,w,num_qubits,n_shots, backend, qiskitService):
    circuits = getProjCircuitsHG(num_qubits, data, w)
    getGatesDistro(circuits, backend)
    simResults_job_id = tr.sendBatch(circuits, n_shots, backend)
    results = getResultsBatch(simResults_job_id, qiskitService)
    np.savetxt('batchResHG.csv', results, delimiter=',')
    return results



def getLargeCircuit(data, w, num_qubits, n_shots, backend, qiskitService):
    # This method is going to append sub circuits into a large circuit
    rows, cols = np.shape(data)
    # rows: number of P's
    # cols: Data
    '''
    Data of the backend:
    '''
    nq_backend = backend.num_qubits
    nq_backend = num_qubits*cols # Patch, remove
    '''
    Quantum Circuit 
    '''

    N = nq_backend
    Nc = cols # I need one bit per ancilliar qubit
    qr = QuantumRegister(N - cols, 'q')
    ar = QuantumRegister(cols, 'ancilla') #And an ancilliar qubit per each data
    cr = ClassicalRegister(Nc, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    '''
    Subcircuit
    '''
    nq_min= num_qubits
    nq_min_use = num_qubits - 1
    capacity = np.floor(nq_backend / num_qubits)
    total_aq = cols

    aqbits = np.arange(nq_min_use * cols, nq_min * cols).tolist()

    
    psi_plus = np.ones(2**(nq_min_use)) # Individual state
    
    for i in range(cols):
        # I need to make a specific target for each data
        target = np.arange(i * nq_min_use, (i + 1) * nq_min_use).tolist()
        # Initialize in | + >
        
        qc.h(target)

        # Unitary transforms 
        uiMat = getUiSimple(data[:,i])
        uwMat = getUiSimple(w)
        print('Gate shape', np.shape(uiMat))
        #print(uiMat)
        #print(uwMat)
        np.savetxt('ui.txt', uiMat)
        # Transform them into unitary gates 
        ui = UnitaryGate(uiMat)
        uw = UnitaryGate(uwMat)

        # Add them to the circuit (build the perceptron) 
        qc.append(ui, target)
        qc.append(uw, target)
        qc.h(target)
        qc.x(target)

        # Entangle the qubits with the ancilla qubit 
        qc.mcx(target, aqbits[i])
        # Measure the ancilla qubit 
        qc.measure(aqbits[i], i) #ancilliar, classical bit -> cols
    
    print(qc)
    return qc

def getLargeCircuitHG(data, w, num_qubits, n_shots, backend, qiskitService):
    # This method is going to append sub circuits into a large circuit
    rows, cols = np.shape(data)
    # rows: number of P's
    # cols: Data
    '''
    Data of the backend:
    '''
    nq_backend = backend.num_qubits
    nq_backend = num_qubits*cols # Patch, remove
    '''
    Quantum Circuit 
    '''

    N = nq_backend
    Nc = cols # I need one bit per ancilliar qubit
    qr = QuantumRegister(N - cols, 'q')
    ar = QuantumRegister(cols, 'ancilla') #And an ancilliar qubit per each data
    cr = ClassicalRegister(Nc, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    '''
    Subcircuit
    '''
    nq_min= num_qubits
    nq_min_use = num_qubits - 1
    capacity = np.floor(nq_backend / num_qubits)
    total_aq = cols

    aqbits = np.arange(nq_min_use * cols, nq_min * cols).tolist()

    
    psi_plus = np.ones(2**(nq_min_use)) # Individual state
    
    for i in range(cols):
        # I need to make a specific target for each data
        target = np.arange(i * nq_min_use, (i + 1) * nq_min_use).tolist()
        
        # Initialize in | + >
        qc.h(target)

        # Hypergraphs algorithms
        qc = uiHG(qc, cols)
        qc = uwHG(qc, w)
        
        # Entangle the qubits with the ancilla qubit 
        qc.mcx(target, aqbits[i])
        # Measure the ancilla qubit 
        qc.measure(aqbits[i], i) #ancilliar, classical bit -> cols
    
    print(qc)
    return qc

