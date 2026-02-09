import numpy as np 
import matplotlib.pyplot as plt
#import pandas as pd
import transition as sit
import scipy.io as sio
import bitarray as bta
import bitarray.util as butil
import transforms as tr
#from scipy.stats import moment
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
#from mlutils import subset_split
from sklearn.model_selection import train_test_split


#QuantumComputing 
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.circuit.library import CZGate
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import matplotlib.colors as colors
import matplotlib.cm as cm
from qiskit.circuit.library import MCXGate, UnitaryGate
from HouseHolder import getUnitaryHH




def scale(database, resolution, max_value):
    '''
    Function to scale a database to a given bits resolution. 
    Parameters: Database to scale, the desired resolution, the maximum value the database can reach. 
    Returns: The database scaled. 
    '''
    for i in range(len(database)):
        database[i] = np.floor(resolution * database[i] / max_value )
    return database


def getBars(meanPhotonNumber, datapoints):
    '''
    Loads a database with the specified parameters.
    Parameters: Mean photon number, datapoints (resolution)
    Returns: A python dictionary 
    '''
    path = '/home/guerrero/Documents/UNAM/7TH-SEMESTER/ICN/References/Database/Datos_Histogramas/'
    name = 'H'+str(meanPhotonNumber)+'_M_'+str(datapoints)+'.mat'
    return sio.loadmat(path+name)


def scatterPlot(database, limit, title): 
    '''
    Displays a scatterPlot. Note: This method is not generic. 
    Parameters: A dictionary database, the limit of ploting the data and a title for the plot.
    Returns: None
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(database['Coh'][0][:limit], database['Coh'][1][:limit], database['Coh'][2][:limit], c='r', marker = '*')
    ax.scatter(database['Th'][0][:limit], database['Th'][1][:limit], database['Th'][2][:limit], c = 'b', marker = '*')
    plt.title(title) 
    plt.legend()
    #plt.show()

'''
TODO: Create a method that makes a histogram of the new data. 
1. Create a hash function to indexing the new bitarray, remember that the method bitarray.to01() returns a string representing the bitarray, you can use this for converting the string into an integer and create a new database to make the histogram.

'''


def bars2bin(database, numBits):
    '''
    Converts a database to it's representation in binary, using binary arrays from bitarray. 
    Parameters: a database 
    Returns: A vectorized database with all the entries of the original database converted to binary. 
    '''
    #print(np.shape(database))
    vecBin = bta.bitarray()
    #print('initial:', vecBin)
    decBitarray = np.zeros(len(database[0]), dtype = np.int64)
    
    for i in range(len(database[0])):
        for j in range(len(database)):
            for k in np.binary_repr(database[j][i], numBits):
                # The bitarray object needs to be filled one by one bit. 
                vecBin.append(int(k))
            #print('loc:', i,',',j,' = ',vecBin, '\n')
        #print(i*len(database)*numBits)
        #print('vecBin:', len(vecBin))
        #print(len(database)*numBits)
        # Here the number containing all the first histogram is stored into a single bits * rows bit number.
        decBitarray[i] = butil.ba2int(vecBin[i*len(database)*numBits:])
        #if(i == 10):exit(1)
    return vecBin, decBitarray

'''
TODO: 
The main goal is to see the difference between the moments of the coeherent and thermal light, so we have this degrees of freedom: 
1. Mean photon Number (4->0.4, 53->0.53, 67->0.67, 77->0.77, 735->0.735) , n = 5
2. Datapoints 10 : 10: 160, n = 16
3. Number of bits 1:1:9 , n = 9 


Strictly speaking, we could plot a surface ploting the diference in the moment with two of the three degrees changing. The main degree is the mean photon number. 
The changing degrees are the datapoints and number of bits, check how to fill the NA's. 

''' 

def plotHistogram(cohData, thData, title):
    print(np.histogram(cohData, bins = 'auto'))
    plt.hist(cohData, alpha = 0.4, bins ='auto')
    plt.hist(thData, alpha = 0.4, bins = 'auto')
    plt.title(title)
    plt.set_xlabel = 'States'
    plt.set_ylabel = 'Observations'
    plt.legend()
    plt.show()


def getMean(meanPhotonNumber, datapoints, numBits):
    '''
    Returns the mean of the coherent and thermal light 
    '''

    # Variables
    resolution = 2**numBits - 1

    # Load the database with the given paramaeters
    distribution = getBars(meanPhotonNumber, datapoints)

    #Plot the original data (optional)
    #scatterPlot(distribution, 500, 'Original data')

    # 1. Scale the database to the given resolution 
    scaledDatabase = {
         'Coh' : scale(distribution['Coh'], resolution, 1),
         'Th' : scale(distribution['Th'], resolution, 1)
     }

    # 1.1 Plot the scaled database (optional)
    scatterPlot(scaledDatabase, 500, 'Scaled data to '+str(numBits)+'bits (' + str(resolution) + ' dec)')
    plt.savefig('scaledCloud'+str(numBits)+'_'+str(meanPhotonNumber)+'_'+str(datapoints))
    plt.close()

    # 2 Transform the data to a binary array 
    cohBin, cohDec = bars2bin(scaledDatabase['Coh'].astype(int), numBits)
    thBin, thDec = bars2bin(scaledDatabase['Th'].astype(int), numBits)

    # 2.1 plot the histograms of the new data 
    plotHistogram(cohDec, thDec, 'Coherent and thermal light with mean photon number ='+str(meanPhotonNumber/100)+' , ' + str(datapoints) + ' datapoints'+', Bits ='+str(numBits))
    plt.savefig('hist'+str(numBits)+'_'+str(meanPhotonNumber)+'_'+str(datapoints))
    plt.close()
    # 3 Return the means
    return np.mean(cohDec), np.mean(thDec)

def getMeanDifference(meanPhotonNumber):
    '''
    TODO: 
    The main goal is to see the difference between the moments of the coeherent and thermal light, so we have this degrees of freedom: 
    1. Mean photon Number (4->0.4, 53->0.53, 67->0.67, 77->0.77, 735->0.735) , n = 5
    2. Datapoints 10 : 10: 160, n = 16
    3. Number of bits 1:1:9 , n = 9 

    '''
    datapointsArr = np.arange(40,170,10) # [10, ... , 160]
    bitsArr = np.arange(1,10,1) # [1, ... , 9]
    meshDifference = np.zeros((len(datapointsArr), len(datapointsArr))) #[[0, ... , 0], ... , [0, ..., 0]]
    
    for dpoint in range(10,len(datapointsArr)): 
        for bits in range(6, len(bitsArr)):
            meanCoh, meanTh = getMean(meanPhotonNumber, datapointsArr[dpoint], bitsArr[bits])
            print(dpoint)
            print(bits)
            meshDifference[dpoint][bits] = np.abs(meanCoh - meanTh)
            print(np.abs(meanCoh - meanTh))

    return meshDifference

def test():

    #Get csv files
    distro = getBars(4, 40)       # Coherent, mean ph. = 0.4, 150 datapoints

    trsObj = sit.transition() # New transition object

    print(distro['Coh'])
    print(distro['Th'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    #ax.scatter()
    
    limit = 500
    

    ax.scatter(distro['Coh'][0][:limit], distro['Coh'][1][:limit], distro['Coh'][2][:limit], c='r', marker = '*')
    ax.scatter(distro['Th'][0][:limit], distro['Th'][1][:limit], distro['Th'][2][:limit], c = 'b', marker = '*')
    plt.title('Original') 
    plt.show()
    

    #Parameters 
    numBits = 8
    #Max resolulution = 9
    resolution = 2**numBits - 1


    scaledDistro = {
        'Coh' : scale(distro['Coh'], resolution, 1),
        'Th' : scale(distro['Th'], resolution, 1)
    }
    

    scatterPlot(scaledDistro, 500, 'scaledDistro')
    print(np.shape(scaledDistro['Coh']))
    print(np.shape(scaledDistro['Th']))

    print(scaledDistro['Coh'])
    


    cohBin, cohDec = bars2bin(scaledDistro['Coh'].astype(int), numBits)
    print(len(cohBin))
    thBin, thDec = bars2bin(scaledDistro['Th'].astype(int), numBits)
    #print(np.shape(cohBin))
    #thBin = bars2bin(scaledDistro['Th'].astype(int))

    #plt.hist(cohBin)
    print(len(cohDec))
    print(cohDec)

    print(thDec)
    
    print('Moment of coh', np.mean(cohDec))
    print('Moment of th', np.mean(thDec))

    plt.hist(cohDec)
    plt.hist(thDec)
    plt.show()

    #(4->0.4, 53->0.53, 67->0.67, 77->0.77, 735->0.735)

    maxvals = []
    maxValsEval = []


    for meanPhNum in np.array([77]):
        print('Begining of the getMeanDifference function ______')
        diffMatrix = getMeanDifference(meanPhNum)
        plt.imshow(diffMatrix, cmap = 'hot', interpolation = 'nearest')
        plt.colorbar()  # Add color bar for reference
        plt.xlabel('Number of bits')
        plt.ylabel('Datapoints')
        plt.title('Difference in mean')
        plt.savefig(str(meanPhNum))
        plt.close()
        #plt.show()
        max_index = np.argmax(diffMatrix)
        max_coord = np.unravel_index(max_index, diffMatrix.shape)
        maxvals.append(max_coord)
        maxValsEval.append(diffMatrix[max_coord[0]][max_coord[1]])

    print(maxvals)
    idex = np.array(maxvals)
    np.save(np.array(maxvals, arr))
    
    plt.plot(maxValsEval)
    plt.save('MaximumDifferences')

    plt.close()
    np.save(np.array(maxvals, arr))

def data2BinDec(meanPhotonNumber, datapoints, numBits):
    # Variables
    resolution = 2**numBits - 1

    # Load the database with the given paramaeters
    distribution = getBars(meanPhotonNumber, datapoints)

    #Plot the original data (optional)
    #scatterPlot(distribution, 500, 'Original data')

    # 1. Scale the database to the given resolution 
    scaledDatabase = {
         'Coh' : scale(distribution['Coh'], resolution, 1),
         'Th' : scale(distribution['Th'], resolution, 1)
     }

    # 2 Transform the data to a binary array 
    cohBin, cohDec = bars2bin(scaledDatabase['Coh'].astype(int), numBits)
    thBin, thDec = bars2bin(scaledDatabase['Th'].astype(int), numBits)
    
    data = {
        'Coh' : [cohBin, cohDec],
        'Th' : [thBin, thDec]
    }


def get_histogram(meanPhotonNumber, datapoints, numBits, ran):
    '''
    Returns a disctionary with the beans and observation of distributions for the coherent and thermal light.
    '''
    # Variables
    resolution = 2**numBits - 1

    # Load the database with the given paramaeters
    distribution = getBars(meanPhotonNumber, datapoints)

    #Plot the original data (optional)
    #scatterPlot(distribution, 500, 'Original data')

    # 1. Scale the database to the given resolution 
    scaledDatabase = {
         'Coh' : scale(distribution['Coh'][ran], resolution, 1),
         'Th' : scale(distribution['Th'][ran], resolution, 1)
     }

    # 2 Transform the data to a binary array 
    cohBin, cohDec = bars2bin(scaledDatabase['Coh'].astype(int), numBits)
    thBin, thDec = bars2bin(scaledDatabase['Th'].astype(int), numBits)

    histograms = {
        'Coh' : np.unique(cohDec, return_counts = True),
        'Th' : np.unique(thDec, return_counts = True)
    }
    return histograms

def getData(numBits, meanPhotonNumber, datapoints):
    '''
    Returns a disctionary with the data of thermal an coherent ligth in binary.
    '''
    # Variables
    resolution = 2**numBits - 1

    # Load the database with the given paramaeters
    distribution = getBars(meanPhotonNumber, datapoints)

    #Plot the original data (optional)
    #scatterPlot(distribution, 500, 'Original data')

    # 1. Scale the database to the given resolution 
    scaledDatabase = {
         'Coh' : scale(distribution['Coh'], resolution, 1),
         'Th' : scale(distribution['Th'], resolution, 1)
     }

    # 2 Transform the data to a binary array 
    cohBin, cohDec = bars2bin(scaledDatabase['Coh'].astype(int), numBits)
    thBin, thDec = bars2bin(scaledDatabase['Th'].astype(int), numBits)
    
    binData = {
        'Coh' : cohBin,
        'Th' : thBin
    }

    decData = {
        'Coh': cohDec,
        'Th' : thDec
    }

    return binData, decData

    

def plot_scaled_histogram(meanPhotonNumber, datapoints, numBits):
    '''
    Returns a disctionary with the beans and observation of distributions for the coherent and thermal light.
    '''
    # Variables
    resolution = 2**numBits - 1

    # Load the database with the given paramaeters
    distribution = getBars(meanPhotonNumber, datapoints)

    #Plot the original data (optional)
    #scatterPlot(distribution, 500, 'Original data')

    # 1. Scale the database to the given resolution 
    scaledDatabase = {
         'Coh' : scale(distribution['Coh'], resolution, 1),
         'Th' : scale(distribution['Th'], resolution, 1)
     }

    # 2 Transform the data to a binary array 
    cohBin, cohDec = bars2bin(scaledDatabase['Coh'].astype(int), numBits)
    thBin, thDec = bars2bin(scaledDatabase['Th'].astype(int), numBits)

    plotHistogram(cohDec, thDec, 'Coherent and thermal light with mean photon number ='+str(meanPhotonNumber/100)+' , ' + str(datapoints) + ' datapoints'+', Bits ='+str(numBits))

    


def zero2minus(zerosArray):
    # Create a byte 
    minusArray = (-1)* (np.ones(len(zerosArray), dtype = np.int8))
    for i in range(len(zerosArray)):
        minusArray[i] = minusArray[i]**(np.int8(zerosArray[i]))
    return minusArray

def beans2bin(binsCoh, binsTh, numBits, numBars):
    '''
    The histogram now is a dictionary with two keys: Coh an Th, each with an array containing the beans and the observations of those beans. I got to extract the states ( beans ). Each state must be a number of numBits * number of rows. In the main example we use 8 bits and 7 rows (bars) so it should be a number of 56 bits. Here I must decide if I'll collapse all the states into a new single number and slice it when needed with a hash function or, for seek of simplicity, just create a matrix or an array of chars, but this would take up to 8 times more memory and processing time. I think I'm going for the first one.  
    '''
    cohStates = bta.bitarray() #Empy bitarray
    thStates = bta.bitarray()
    
    for state in binsCoh: #State is a number
        #print(state)
        for bit in np.binary_repr(np.int64(state), numBits*numBars): #Convert that number into binary and iterate through it 
            cohStates.append(np.int8(bit)) #Append each bit into the bitarray
    
    for state in binsTh:
        for bit in np.binary_repr(np.int64(state), numBits*numBars):
            thStates.append(np.int8(bit))
        #print(thStates, len(thStates))
    
    states = {
        'Coh' : cohStates, 
        'Th' : thStates
    }

    return states



def printStates(states, numBits, numBars, numObs):
    for i in range(numBits * numBars, numBits * numBars * (1 + numObs), numBits * numBars):
        print(states['Th'][i - numBits*numBars:i], len(states['Th'][i - numBits*numBars:i]))

def proj2v(states, vState, numBits, numBars, numObs, norm):
    prod = 0 
    #proj = [] # Change to an array, latter
    proj = np.zeros(numObs)
    iter = 0
    for i in range(numBits * numBars, numBits * numBars * (1 + numObs), numBits * numBars):
        prod = 0
        #print(states[i - numBits*numBars:i])
        for j in range(len(states[i - numBits*numBars:i])):
            prod += (states[i - numBits*numBars:i][j]*vState[j])
        #proj.append(prod)
        proj[iter] = (1/norm)*prod
        iter += 1
        '''
        if(i <= numBits * numBars * 3):
            print('MFS:', vState)
            print('State:', states[i - numBits*numBars:i])
            print('Proj:', proj)
        '''
    return np.array(proj)

def normalize(state, numBits, numBars, numObs):
    n =  np.sqrt(proj2v(state, state, numBits, numBars, numObs, numBits*numBars))
    return n * state

def projectionToMFS(statesBin, binMFS, numBits, numBars, numObs):

    norm = numBits*numBars
    projsCoh = proj2v(statesBin['Coh'], binMFS , numBits, numBars, numObs['Coh'],norm)

    projsTh = proj2v(statesBin['Th'], binMFS , numBits, numBars, numObs['Th'], norm)

    projsCohP = proj2v(statesBin['Coh'], binMFS , numBits, numBars, numObs['Coh'], norm)

    projsThP = proj2v(statesBin['Th'], binMFS , numBits, numBars, numObs['Th'], norm)

    

    print(mostFreqState)
    print((histDic['Coh'][1]), len(histDic['Coh'][0]))
    plt.plot(histDic['Coh'][0], histDic['Coh'][1])
    plt.title('Histogram with all the states')
    plt.xlabel('States (decimal)')
    plt.ylabel('Frequency')
    plt.show()

    #Get the histogram for the Projections
    print(len(projsCoh))
    print(len(projsTh))
    projsHist = {
        'Coh' : np.unique(projsCoh, return_counts = True),
        'Th' : np.unique(projsTh, return_counts = True),
        'CohP' : np.unique(projsCohP, return_counts = True),
        'ThP' : np.unique(projsThP, return_counts = True)
    }

    print(projsHist)

    #plt.bar(np.array([f'{num:.3g}' for num in statesCoh], dtype = str), np.array(projsCoh))
    
    #print((projsHist['Coh'][1]), len(projsHist['Coh'][mostFreqState0]))
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.bar(projsHist['Coh'][0], projsHist['Coh'][1], alpha = 0.4)
    plt.bar(projsHist['Th'][0], projsHist['Th'][1], alpha = 0.4)
    plt.title('Projection of coherent and thermal states with their most frequent state')
    plt.xlabel('Projection with the | M_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| M_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')

    plt.subplot(1,2,2)
    plt.bar(projsHist['CohP'][0], projsHist['CohP'][1], alpha = 0.4)
    plt.bar(projsHist['ThP'][0], projsHist['ThP'][1], alpha = 0.4)
    plt.title('Projection of coherent and thermal states with crossed most frequent state')
    plt.xlabel('Projection with the | M_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| M_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def getNegProjMFS(meanPhNum, datapoints, numBits, numBars, ran):
    norm = numBits*numBars

    # Get the histogram of the desired database with the given parameters. 
    histDic = get_histogram(meanPhNum, datapoints, numBits, ran)
    #plot_scaled_histogram(meanPhNum,datapoints,numBits)
    # Now we have a dictionary with the frequencies and their bean edges
    # We'll need the number of observations 
    numObs = {
        'Coh' : len(histDic['Coh'][0]), 
        'Th' : len(histDic['Th'][0])
    }
    
    mostFreqState = {
        'Coh' : histDic['Coh'][0][np.argmax(histDic['Coh'][1])],
        'Th' : histDic['Th'][0][np.argmax(histDic['Th'][1])],
    }

    # To do the projection, first we need to get the midpoint of the bean edges in decimal
    # Now we convert all the beans into a single bitarray of dimensions numBits * numRows * numObs
    statesBin = beans2bin(histDic['Coh'][0], histDic['Th'][0], numBits, numBars)


    statesInOnes = {
        'Coh' : zero2minus(statesBin['Coh']), 
        'Th' : zero2minus(statesBin['Th'])
    }



    meansInOnes = {
        'Coh' : zero2minus(bta.bitarray(np.binary_repr(np.int64(mostFreqState['Coh']), numBits*numBars))), 
        'Th' : zero2minus(bta.bitarray(np.binary_repr(np.int64(mostFreqState['Th']), numBits*numBars))), 
    }

    
    
    projInOnes = {
        'Coh' : proj2v(statesInOnes['Coh'], meansInOnes['Coh'], numBits, numBars, numObs['Coh'], norm), # Proj of coheret with coherent repr
        'CohP': proj2v(statesInOnes['Coh'], meansInOnes['Th'], numBits, numBars, numObs['Coh'], norm), #Proj of coherent with thermal repr
        'Th' : proj2v(statesInOnes['Th'], meansInOnes['Th'], numBits, numBars, numObs['Th'], norm), # Proj of thermal with thermal repr
        'ThP' : proj2v(statesInOnes['Th'], meansInOnes['Coh'], numBits, numBars, numObs['Th'], norm) # Proj of thermal in coherent repr
    }

    
    negativeProj = {
        'Coh' : np.where(projInOnes['Coh'] < 0),
        'Th' : np.where(projInOnes['Th'] < 0),
        'CohP' : np.where(projInOnes['CohP'] < 0),
        'ThP' : np.where(projInOnes['ThP'] < 0), 
    }
    return (negativeProj)

def printNumNegProj(meanPhNum, datapointsVec, numBits, numBars):
    for dp in datapointsVec:
        print(str(dp))
        print(getNegProjMFS(meanPhNum,dp,numBits,numBars))


def modWeightVec(mfs, errorVect):
    print(errorVect)
    mfsTmp = mfs
    for i in range(len(errorVect)):
        for n in range(len(mfs)):
            mfsTmp[n] = mfsTmp[n] - 2*(mfsTmp[n])*(errorVect[i]**2)
    # Prit the new weightVec
    print(mfsTmp)
    return mfsTmp

def predict(binData, decData, numBits, numBars, weightVec, treshold, numEpoch):
    # Get the histogram of the desired database with the given parameters. 


    numCoh = len(decData['Coh'])
    numTh = len(decData['Th'])
    numObs = numCoh + numTh
    

    # Preparation of the predictions vector 
    predictions = np.zeros((numEpoch, numObs), dtype = np.int8)
    # correct classification vector 
    classVect = np.zeros(numObs, dtype = np.int8)
    errorVect = np.zeros(numObs, dtype = np.int8)
    mseErrorVect = np.zeros(numEpoch) 


    print('Num of Coherent sates: ', numCoh)
    print('Num of Thermal states:', numTh)
    print('Sum: ', numObs)
    #Zero is thermal and 1 is coherent

    '''
    Now we mix all the states 
    '''
    allData = binData['Coh']


    for i in binData['Th']:
        allData.append(i)
    print(len(allData))
    allData = zero2minus(allData)
    #plt.plot(projections)
    #plt.show()
    #print(projections)
    

    '''
    Classification: 
    If the source is thermal ( < 28), the element of the predicition vector remains as zero , if is coherent (>= 28), a 1 is assigned tohe element. 
    '''
    # Creation of the clssVector, this vector stores the correct classification of the data. It plays the role of y 
    # while predictions is the y hat vector. 
    # The data is appended so we can use the order to keep the classification 
    # This may need to be changed for another way of keeping the classificacion

    classVect[numTh:] = 1 # The element after the number of thermal observations are all coherent. 
    print('OriginalWeightVec:',weightVec)
    for epoch in range(numEpoch):
        # Iterate over all epochs
        # Now me make the projections 
        print('NewWeight Vec:', weightVec)
        projections = proj2v(allData, weightVec, numBits, numBars, numObs, norm)

        for i in range(len(projections)):
            # Activation function
            if(projections[i] >= treshold):
                predictions[epoch][i] = 1
            # Calculate error
            errorVect[i] = classVect[i] - predictions[epoch][i]

            for n in range(len(weightVec)):
                print('wn:', weightVec[n])
                print('error:', errorVect[i])
                print('2*wn*error**2:', 2*(weightVec[n])*(errorVect[i]**2))
                weightVec[n] = weightVec[n] - 2*(weightVec[n])*(errorVect[i]**2)
                print('wn+1:', weightVec[n])
                print(weightVec)
                if(n == 6): exit()
            #print('Inside:', weightVec)

        #print('weightVec:',weightVec)
        #print('errorVect:',errorVect)
        mseErrorVect[epoch] = np.mean(errorVect**2)
        print(mseErrorVect[epoch])
        #print(predictions)

    return errorVect, mseErrorVect, weightVec



def perceptron(allData, lenData, numBits, numBars, weightVec, treshold):
    # Get the histogram of the desired database with the given parameters. 


    numCoh = lenData['Coh']
    numTh = lenData['Th']
    numObs = numCoh + numTh

    # Preparation of the predictions vector 
    predictions = np.zeros(numObs, dtype = np.int8)
    # correct classification vector 
    classVect = np.zeros(numObs, dtype = np.int8)
    errorVect = np.zeros(numObs, dtype = np.int8)
    mseError = 0 


    '''
    Classification: 
    If the source is thermal ( < 28), the element of the predicition vector remains as zero , if is coherent (>= 28), a 1 is assigned tohe element. 
    '''
    # Creation of the clssVector, this vector stores the correct classification of the data. It plays the role of y 
    # while predictions is the y hat vector. 
    # The data is appended so we can use the order to keep the classification 
    # This may need to be changed for another way of keeping the classificacion

    classVect[numTh:] = 1 # The element after the number of thermal observations are all coherent. 
    #print('OriginalWeightVec:',weightVec)
    
    # Iterate over all epochs
    # Now me make the projections 
    #print('NewWeight Vec:', weightVec)
    projections = proj2v(allData, weightVec, numBits, numBars, numObs, 1) 

    for i in range(len(projections)):
        # Activation function
        if(projections[i] >= treshold):
            predictions[i] = 1
        # Calculate error
        errorVect[i] = classVect[i] - predictions[i]


    #print('weightVec:',weightVec)
    #print('errorVect:',errorVect)
    mseError = np.mean(errorVect**2)
    #print(predictions)

    return errorVect, mseError

def testProjections():

    '''
    1. Mean photon Number (4->0.4, 53->0.53, 67->0.67, 77->0.77, 735->0.735) , n = 5
    2. Datapoints 10 : 10: 160, n = 16
    3. Number of bits 1:1:9 , n = 9 

    '''
    numBits = 8
    numBars = 7
    meanPhNum = 77
    datapoints = 160 
    norm = numBits*numBars
    #################################33
    ran  = range(4, 7)
    numBars = numBars - len(ran)

    numBitsVec = np.arange(1,numBits + 1, 1)
    numBarsVec = np.arange(3, numBars + 1, 1)
    meanPhNumVec = np.array([4, 53, 67, 77, 735])
    datapointsVec = np.arange(30, 170, 10)

    # Get the histogram of the desired database with the given parameters. 
    histDic = get_histogram(meanPhNum, datapoints, numBits, ran)
    plot_scaled_histogram(meanPhNum,datapoints,numBits)


    #Now we have a dictionary with the frequencies and their bean edges
    #histogram[Coherent or thermal][0: frequencies, 1: bean edges]
    print('Bin edges:',len(histDic['Coh'][0]),' States:', len(histDic['Coh'][1]))
    
    # We'll need the number of observations 
    numObs = {
        'Coh' : len(histDic['Coh'][0]), 
        'Th' : len(histDic['Th'][0])
    }
    
    # Create a dictionary with the means of the thermal and coherent h

    
    mostFreqState = {
        'Coh' : histDic['Coh'][0][np.argmax(histDic['Coh'][1])],
        'Th' : histDic['Th'][0][np.argmax(histDic['Th'][1])],
    }

    # To do the projection, first we need to get the midpoint of the bean edges in decimal
    #statesCoh, statesTh = getBeans(histDic)
    #print(statesCoh, statesTh)

    # Now we convert all the beans into a single bitarray of dimensions numBits * numRows * numObs
    statesBin = beans2bin(histDic['Coh'][0], histDic['Th'][0], numBits, numBars)
    
    # Plot 
    
    projsCoh = proj2v(statesBin['Coh'], bta.bitarray(np.binary_repr(np.int64(mostFreqState['Coh']), numBits*numBars)), numBits, numBars, numObs['Coh'], norm)

    projsTh = proj2v(statesBin['Th'], bta.bitarray(np.binary_repr(np.int64(mostFreqState['Th']), numBits*numBars)), numBits, numBars, numObs['Th'], norm)

    projsCohP = proj2v(statesBin['Coh'], bta.bitarray(np.binary_repr(np.int64(mostFreqState['Th']), numBits*numBars)), numBits, numBars, numObs['Coh'], norm)

    projsThP = proj2v(statesBin['Th'], bta.bitarray(np.binary_repr(np.int64(mostFreqState['Coh']), numBits*numBars)), numBits, numBars, numObs['Th'], norm)

    

    print(mostFreqState)
    #print((histDic['Coh'][1]), len(histDic['Coh'][0]))
    #plt.plot(histDic['Coh'][0], histDic['Coh'][1])
    #plt.title('Histogram with all the states')
    #plt.xlabel('States (decimal)')
    #plt.ylabel('Frequency')
    #plt.show()

    #Get the histogram for the Projections
    print(len(projsCoh))
    print(len(projsTh))
    projsHist = {
        'Coh' : np.unique(projsCoh, return_counts = True),
        'Th' : np.unique(projsTh, return_counts = True),
        'CohP' : np.unique(projsCohP, return_counts = True),
        'ThP' : np.unique(projsThP, return_counts = True)
    }

    #print(projsHist)

    #plt.bar(np.array([f'{num:.3g}' for num in statesCoh], dtype = str), np.array(projsCoh))
    
    #print((projsHist['Coh'][1]), len(projsHist['Coh'][mostFreqState0]))
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.bar(projsHist['Coh'][0], projsHist['Coh'][1], width = 0.01, alpha = 0.4, color = "blue", label = 'Coh-CohMFS')
    plt.bar(projsHist['Th'][0], projsHist['Th'][1], width = 0.01, alpha = 0.4, color = "red", label = 'Th-ThMFS')
    # Creating a custom legend with colors
    legend_labels = ['Coh-CohMFS', 'Th-ThMFS']
    legend_colors = ["blue", "red"]
    custom_legend = [plt.Rectangle((0,0), 1, 1, color=color) for color in legend_colors]
    plt.legend(custom_legend, legend_labels)
    plt.title('Projection of coherent and thermal states(binary) with their most frequent state')
    plt.xlabel('Projection with the | M_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| M_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')

    plt.subplot(1,2,2)
    plt.bar(projsHist['CohP'][0], projsHist['CohP'][1], width = 0.01, alpha = 0.4,  color = "blue", label = 'Coh-ThMFS')
    plt.bar(projsHist['ThP'][0], projsHist['ThP'][1], width = 0.01, alpha = 0.4,  color = "red", label = 'Th-CohMFS')
    legend_labels = ['Coh-thMFS', 'Th-CohMFS']
    legend_colors = ["blue", "red"]
    custom_legend = [plt.Rectangle((0,0), 1, 1, color=color) for color in legend_colors]
    plt.legend(custom_legend, legend_labels)

    plt.title('(BINARY) Projection of coherent and thermal states with crossed MFS')
    plt.xlabel('Projection with the | MFS_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| MFS_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


    


    '''
    Now I'll do the same buth with ones and minus ones 
    TODO: 
    1. Find a way to extract the most repeated state in the distribution 
    2. Take the projection of each state into the max state 

    '''
    


    

    statesInOnes = {
        'Coh' : zero2minus(statesBin['Coh']), 
        'Th' : zero2minus(statesBin['Th'])
    }

    #print('states', statesInOnes, len(statesInOnes['Coh']))
    #printStates(statesInOnes, numBits, numBars, numObs)
    #printStates(statesInOnes['Th'], numBits, numBars, numObs)

    meansInOnes = {
        'Coh' : zero2minus(bta.bitarray(np.binary_repr(np.int64(mostFreqState['Coh']), numBits*numBars))), 
        'Th' : zero2minus(bta.bitarray(np.binary_repr(np.int64(mostFreqState['Th']), numBits*numBars))), 
    }
    #print('means', meansInOnes, len(meansInOnes['Coh']),',', len(meansInOnes['Th']))

    print('Bin:', bta.bitarray(np.binary_repr(np.int64(mostFreqState['Coh']), numBits*numBars)))
    print('Ones:', meansInOnes['Coh'])
    print(len(bta.bitarray(np.binary_repr(np.int64(mostFreqState['Coh']), numBars*numBits))))


    
    print(len(meansInOnes['Coh']))
    projInOnes = {
        'Coh' : proj2v(statesInOnes['Coh'], meansInOnes['Coh'], numBits, numBars, numObs['Coh'], numBits*numBars), # Proj of coheret with coherent repr
        'CohP': proj2v(statesInOnes['Coh'], meansInOnes['Th'], numBits, numBars, numObs['Coh'], numBits*numBars), #Proj of coherent with thermal repr
        'Th' : proj2v(statesInOnes['Th'], meansInOnes['Th'], numBits, numBars, numObs['Th'], numBits*numBars), # Proj of thermal with thermal repr
        'ThP' : proj2v(statesInOnes['Th'], meansInOnes['Coh'], numBits, numBars, numObs['Th'], numBits*numBars) # Proj of thermal in coherent repr
    }



    projsHistOnes = {
        'Coh' : np.unique(projInOnes['Coh'], return_counts = True), 
        'Th' : np.unique(projInOnes['Th'], return_counts = True),
        'CohP' : np.unique(projInOnes['CohP'], return_counts = True),
        'ThP' : np.unique(projInOnes['ThP'], return_counts = True)
    }

    #print(projsHist)

    #print((projsHist['Coh'][1]), len(projsHist['Coh'][mostFreqState0]))
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.bar(projsHistOnes['Coh'][0], projsHistOnes['Coh'][1], width = 0.01, alpha = 0.4)
    plt.bar(projsHistOnes['Th'][0], projsHistOnes['Th'][1], width = 0.01,alpha = 0.4)
    plt.title('Projection of coherent and thermal states in ones with their most frequent state')
    plt.xlabel('Projection with the | M_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| M_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')

    plt.subplot(1,2,2)
    plt.bar(projsHistOnes['CohP'][0], projsHistOnes['CohP'][1],  width = 0.01,alpha = 0.4)
    plt.bar(projsHistOnes['ThP'][0], projsHistOnes['ThP'][1],  width = 0.01,alpha = 0.4)
    plt.title('Projection of coherent and thermal states with crossed most frequent state')
    plt.xlabel('Projection with the | M_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| M_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
    # Coh - CohRepr , Th - CohRepr ; Coh - ThRepr , Th - ThRepr  
    print(np.where(projInOnes['Coh'] < 0))
    print(np.where(projInOnes['Th'] < 0))
    print(np.where(projInOnes['CohP'] < 0))
    print(np.where(projInOnes['ThP'] < 0))

    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.bar(projsHistOnes['Coh'][0], projsHistOnes['Coh'][1], alpha  = 0.4,  width = 0.01,color = "blue", label = 'Coh-CohMFS')
    plt.bar(projsHistOnes['ThP'][0], projsHistOnes['ThP'][1], alpha = 0.4,  width = 0.01,color = "red", label = 'Th-CohMFS')
    # Creating a custom legend with colors
    legend_labels = ['Coh-CohMFS', 'Th-CohMFS']
    legend_colors = ["blue", "red"]
    custom_legend = [plt.Rectangle((0,0), 1, 1, color=color) for color in legend_colors]
    plt.legend(custom_legend, legend_labels)
    plt.title('Projection of coherent and thermal states in ones with their most frequent state')
    plt.xlabel('Projection with the | MFS_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| MFS_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')
    
    plt.subplot(1,2,2)
    plt.bar(projsHistOnes['CohP'][0], projsHistOnes['CohP'][1], alpha = 0.4,  width = 0.01,color = "blue", label='Coh-Th')
    plt.bar(projsHistOnes['Th'][0], projsHistOnes['Th'][1], alpha = 0.4,  width = 0.01,color="red", label='Th-Th')
    legend_labels = ['Coh-ThMFS', 'Th-ThMFS']
    legend_colors = ["blue", "red"]
    custom_legend = [plt.Rectangle((0,0), 1, 1, color=color) for color in legend_colors]
    plt.legend(custom_legend, legend_labels)

    plt.title('Projection of coherent with thermal MFS and Thermal with coherent MFS')
    plt.xlabel('Projection with the | MFS_coh > = ' + str(mostFreqState['Coh']) + ' and ' + '| MFS_th > = ' + str(mostFreqState['Th']))
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    #printNumNegProj(meanPhNum, datapointsVec, numBits, numBars)
    negProjsCoh = np.zeros(len(datapointsVec))
    negProjsThP = np.zeros(len(datapointsVec))
    for dp in range(len(datapointsVec)):
        coh = getNegProjMFS(meanPhNum, datapointsVec[dp], numBits, numBars, ran)['Coh']
        th = getNegProjMFS(meanPhNum, datapointsVec[dp], numBits, numBars, ran)['ThP']
        print('Negative projections Coh: ',coh)
        print('Negative projection ThP:', th)

        negProjsCoh[dp] = len(coh)
        negProjsThP[dp] = len(th)

    plt.plot(negProjsCoh)
    plt.plot(negProjsThP)
    plt.show()

def learningRate(weightVec, errorValue):
    for n in range(len(weightVec)):
        #print('wn:', weightVec[n])
        #print('error:', errorValue)
        #print('2*wn*error**2:', 2*(weightVec[n])*(errorValue**2))
        weightVec[n] = weightVec[n] - 2*(weightVec[n])*(errorValue**2)
        #print('wn+1:', weightVec[n])
        #print('weightValue', weightVec)
        #if(n == 5): return weightVec
    #print(weightVec)
    return weightVec

    #print('Inside:', weightVec)


def singleProj(i, states, vState, numBits, numBars, norm):
    prod = 0 
    #proj = [] # Change to an array, latter
    #print(states[i - numBits*numBars:i])
    for j in range(len(states[i - numBits*numBars:i])):
        prod += (states[i - numBits*numBars:i][j]*vState[j])
    #proj.append(prod)
    return (1/norm)*prod

def toMatrix(data, numBits, numBars):
    
    numObs = int(len(data)/(numBits*numBars))
    newMatrix = np.zeros((numBits*numBars, numObs))
    print(numObs)
    print(numBits)
    print(numBars)
    c = -1
    for i in range(numObs*numBits*numBars):
        #print(i%(numBits*numBars))
        newMatrix[i%(numBits*numBars)][c] = data[i]
        c += 1*(i%(numBits*numBars) == 0)
    return newMatrix

def quantumProd(qc, mfs, psi_i, nshots):
        
    #Creando circuito 
    # Adding the 1 amplitudes in the qubits
    #print(len(filler))
    #print(psi_i)
    #print(len(psi_i))
    
    ''' Quantum Circuit: Preparation '''
    qc.clear()
    N = qc.num_qubits
    psi_plus = np.ones(2**(qc.num_qubits - 1))

    # Hadammard
    qc.h(np.arange(N - 1).tolist())
    
    # Creamos las transformaciones unitarias 
    
    # Ui
    uiMat = getUnitaryHH(psi_plus, psi_i)
    #print(uiMat)
    # Ancilla
    uiMatAnc = np.kron(np.eye(2), uiMat)
    
    # Add it to the circuit
    ui = UnitaryGate(uiMatAnc)
    target = np.arange(N).tolist()
    qc.append(ui, target)
    #print('StVec for psi_i=', psi_i)
    #print(Statevector(qc))
    
    # Uw 
    
    uwMat = getUnitaryHH(mfs, psi_plus)

    # Ancilla
    uwMatAnc = np.kron(np.eye(2), uwMat)
    
    # Add it to the circuit
    uw = UnitaryGate(uwMatAnc)
    qc.append(uw, target)
    
    # Contract the state to | 0 > tensor N 
    qc.h(np.arange(N - 1).tolist())
    # Now flip it to | 1 > tensor N 
    qc.x(np.arange(N - 1).tolist())

    # Entangle with the ancilla
    qc.mcx(np.arange(N - 1).tolist(), N - 1)
    
    qc.measure(N - 1,0)
    #print(qc)

    '''Simulating the circuit'''
    q_prod_test = np.sqrt(tr.simulate(qc, nshots)['01'] / nshots)

    return q_prod_test





def train():

    '''
    
█▀▄ ▄▀█ ▀█▀ ▄▀█   █▀█ █▀█ █▀▀ █▀█ ▄▀█ █▀█ ▄▀█ ▀█▀ █ █▀█ █▄░█
█▄▀ █▀█ ░█░ █▀█   █▀▀ █▀▄ ██▄ █▀▀ █▀█ █▀▄ █▀█ ░█░ █ █▄█ █░▀█
    '''
    numBits = 8
    numBars = 7
    meanPhNum = 77
    datapoints = 160 

    numBitsVec = np.arange(1,numBits + 1, 1)
    numBarsVec = np.arange(3, numBars + 1, 1)
    meanPhNumVec = np.array([4, 53, 67, 77, 735])
    datapointsVec = np.arange(30, 170, 10)
    numEpochs = 10
    norm = numBits*numBars



    ran = range(4,7)
    #numBars = numBars - len(ran)
    print('numbits * numBars =', numBits, '*', numBars, '=', numBits*numBars)
    
    binData, decData = getData(numBits, meanPhNum, datapoints)
    numObs = len(decData['Coh']) + len(decData['Th']) 
    # Recalling the coherent MFS is 35000895515131904 
    decMFS = np.int64(35000895515131904)
    #decMFS = np.int64(65536)
    #decMFS = np.int64(0)
    print(numBits*numBars)
    binMFS = bta.bitarray(np.binary_repr(decMFS, numBits*numBars))
    print('len of binMFS: ', len(binMFS))
    weightVec = zero2minus(binMFS)
    weightHistoryVec = np.zeros((numEpochs, len(weightVec)), dtype=np.int8)


    #error, mseError, weightTrained = predict(binData, decData, numBits, numBars, onesMFS, 28, 15)

    #mseErrorVec = np.zeros(numEpochs)
    #errorVect = np.zeros(numObs)
    

    '''
    Training and testing set 
    '''
    train_percentage = 0.70
    print(len(decData['Coh']))
    print(len(binData['Coh']))

    #cohTrain, cohTest = subset_split(zero2minus(binData['Coh']), train_percentage, numBits, numBars, len(decData['Coh']))
    
    #thTrain, thTest = subset_split(zero2minus(binData['Th']), train_percentage, numBits, numBars, len(decData['Th']))

    x = toMatrix(zero2minus(binData['Coh']), numBits, numBars).T
    y = toMatrix(zero2minus(binData['Th']), numBits, numBars).T
    
    testSize = 1 - train_percentage 

    cohTrain, cohTest, thTrain, thTest =  train_test_split(x, y, test_size = testSize, random_state = 42)
    
    print('len of test before val:')
    print('c:', np.shape(cohTest))
    print('t', np.shape(thTest))

    valCohSize = int(len(cohTest)/2)
    cohVal = cohTest[valCohSize:]
    cohTest = np.delete(cohTest, np.arange(valCohSize), axis = 0)

    valThSize = int(len(thTest)/2)
    thVal = thTest[valThSize:]
    thTest = np.delete(thTest, np.arange(valThSize), axis = 0)

    
    print('Shape of test after val:')
    print('c:', np.shape(cohTest))
    print('t', np.shape(thTest))

    

    '''
    Now we mix all the states 
    '''
    #Train
    allDataTrain = np.concatenate((cohTrain, thTrain), axis = 0)

    #Test
    allDataTest = np.concatenate((cohTest, thTest), axis = 0)

    #Valitation
    allDataVal = np.concatenate((cohVal, thVal), axis = 0)
    
    # The data should look like a step function
    classificationTrain = np.zeros(len(allDataTrain))
    classificationTest = np.zeros(len(allDataTest))

    classificationVal = np.zeros(len(allDataVal))
    
    # 0 is thermal, 1 is coherent so 
    classificationTrain[0:len(cohTrain)] = 1
    classificationTest[0:len(cohTest)] = 1
    classificationVal[0:len(cohVal)] = 1

    print('Before shuffle')
    print(np.shape(classificationTrain))
    print(np.shape(classificationTest))
    print(np.shape(classificationVal))

    #Shuffle the data
    
    # Create idex
    permuted_id_train = np.random.permutation(len(allDataTrain))
    permuted_id_test = np.random.permutation(len(allDataTest))
    permuted_id_val = np.random.permutation(len(allDataVal))
    
    # Do permutation
    allDataTrain = allDataTrain[permuted_id_train]
    allDataTest = allDataTest[permuted_id_test]
    allDataVal = allDataVal[permuted_id_val]
    
    # Make the canges in classification
    classificationTrain = classificationTrain[permuted_id_train]
    classificationTest = classificationTest[permuted_id_test]
    classificationVal = classificationVal[permuted_id_val]
    
    '''
    Danger zone: I got to change the length of the data so I can test the code faster, deactivate in action.
    '''
    #'''
    #********** DANGER ************#

    subs  = 0.980

    # Do permutation
    allDataTrain = allDataTrain[int(len(allDataTrain)*subs):]
    allDataTest = allDataTest[int(len(allDataTest)*subs):]
    allDataVal = allDataVal[int(len(allDataVal)*subs):]
    
    # Make the canges in classification
    classificationTrain = classificationTrain[int(len(classificationTrain)*subs):]
    classificationTest = classificationTest[int(len(classificationTest)*subs):]
    classificationVal = classificationVal[int(len(classificationVal)*subs):]
  
    #******************************#
    #'''   
    cls_classificationTrain = classificationTrain
    cls_classificationTest = classificationTest
    cls_classificationVal = classificationVal
  
    print('after shuffle')
    print(np.shape(classificationTrain))
    print(np.shape(classificationTest))
    print(np.shape(classificationVal))

    plt.plot(classificationTrain)
    plt.plot(classificationTest)
    plt.plot(classificationVal)
    plt.title('Shuffled classification: Trained, Test, Validation')
    plt.show()

    '''
        
    ▀█▀ █▀█ ▄▀█ █ █▄░█ █ █▄░█ █▀▀   █▀█ █░█ ▄▀█ █▀ █▀▀
    ░█░ █▀▄ █▀█ █ █░▀█ █ █░▀█ █▄█   █▀▀ █▀█ █▀█ ▄█ ██▄

    '''


    predictions_train = np.zeros_like(classificationTrain)
    predictions_test = np.zeros_like(classificationTest)
    predictions_val = np.zeros_like(classificationVal)
    # Classical prediction 
    clsPredictions_train = predictions_train
    clsPredictions_test = predictions_test
    clsPredictions_val = predictions_val
    

    efficiencyArr = []
    clsEfficiencyArr = efficiencyArr
    treshold = 0.075

    tresholdRange = np.array([treshold])
    #tresholdRange = np.arange(0.075, 0.25, 0.075)
    print(tresholdRange)
    #tresholdRange = np.array([treshold])
    #efficiencyTreshold = []
    print(tresholdRange)
            
    mfs = zero2minus(binMFS)
    '''
    Pre trained mfs with 89 accuracy
    '''

    mfs = np.array([ 1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,
  1, -1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1 ,1  ,1  ,1  ,1  ,1 ,-1, 1, 1], dtype = np.int8)

    


    '''
    Quantum Circuit
    '''

    #Creando circuito 
    N = 7
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(2, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    nshots = 100


    norm = numBits*numBars

    # Testing the prior best result
    
    q_prod_test = np.zeros_like(predictions_test)
    q_predictions_test = np.zeros_like(predictions_test)
    disc_prd_t = np.zeros_like(predictions_test)
    # Somme modifications due to the num of qubits

    psi_plus = np.ones(2**(qc.num_qubits - 1), dtype = np.int8)
    
    # Filler section: The residual slots need to be filled 
    fillerAllNeg = np.ones(2**(qc.num_qubits - 1) - numBits*numBars, dtype = np.int8) # Remaining slots are filled with ones 
    # [1 -1 1 -1 -1] [1 1 1 1]
    fillerHalfNeg = fillerAllNeg 
    
    #Adding negative ones 
    fillerHalfNeg[int(len(fillerAllNeg)/2):] = -1 
    # [1 1 -1 -1]
    mfs = np.concatenate((mfs, fillerHalfNeg))
    # The weight is added with the half negative filler, the data will have the all ones filler (fillerMfs)
    mfsClassic = mfs # The bbest result obtanied till now 
    weightHistoryVec = np.zeros((numEpochs, len(mfs)), dtype=np.int8) # A register of the weights all over the training phase 
    

    

    # The filler for the input data will be composed of only ones
        # By convention the mfs will have the decieving half
    #mfs[int(len(fillerMfs)/2):] *= -1
    # To aniquilate the contributions the filler may have
    print(fillerHalfNeg)
    #nshots = 10 #Quantum simulation 
    qc.clear()
    
    '''
    Prior to start the training in quantum simulation, I must try the classical trained weight once again, I made a mistake with the filler, this migth be the reason why the quantum perceptron performed so poorly 

    This extra epoch is for testing the mos efficient weight vector found till now
    '''
    #test_quantum_pre = predictions_test
    treshold = 0.075

    '''
    This test has noting to do with the training phase, this is here to test the best weight found 'till now. The q_predictions_test is a stand alone array as well. 
    '''
    for i in range(len(allDataTest)):
        #Quantum Case
        psi_i = np.concatenate((allDataTest[i], fillerAllNeg)) # Adding filler 
        qprod = quantumProd(qc, mfs, psi_i, nshots) # Performing quantum projection 
        q_predictions_test[i] = 1*(qprod >= treshold) # Make a prediction with the qiuantum projection
        
        #Classical case  
        clsProj_i = mfs@psi_i / (len(mfs)) # Filler is alreagy added, so take the classical product  
        #print(qprod, clsProj_i)
        clsPredictions_test[i] = 1*(clsProj_i >= treshold) # Make a prediction with the projection 
        #print(qprod == clsProj_i)
        #print(1*(qprod >= treshold) ==  1*(clsProj_i >= treshold))
    
    # Now we have predictions for all the data
    # Let's make the confusion matrix for all those predictions
    qECf = confusion_matrix(classificationTest, q_predictions_test)
    # Meassure the efficieny
    qEfficiency = accuracy_score(classificationTest, q_predictions_test)*100
    # efficiencyArr[0] = qEfficiency
    #efficiencyArr = np.append(efficiencyArr, qEfficiency) 
    #weightHistoryVec[0] = mfs

    # Classical ConfusionMatrix
    #print(len(np.where(classificationTest == clsPredictions_test)))
    clscfMatVal = confusion_matrix(cls_classificationTest,clsPredictions_test)
    clsEfficiency = accuracy_score(cls_classificationTest, clsPredictions_test)*100
    #clsEfficiencyArr[0] = clsEfficiency
    #clsEfficiencyArr = np.append(clsEfficiencyArr, clsEfficiency)
    #print((qEfficiency, clsEfficiency))
    #print(efficiencyArr[0], clsEfficiencyArr[0])

    qECfDisp = ConfusionMatrixDisplay(qECf)
    qECfDisp.plot()
    plt.title('Quantum Perceptron classical eff:'+ str(clsEfficiency)  + ',Quantum eff:' + str(qEfficiency))
    plt.savefig('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/MAXCLASSEF-QuantumPerceptron_'+str((train_percentage - 1)/2)+'_numBits'+str(numBits)+str(treshold)+'_treshold_'+str(nshots)+'_nshots.png', dpi=199)

    plt.close()


    '''
    Now training begins 
    '''
    print(mfs)
   
    for treshold in tresholdRange:
        # We can test more tresholds if we want to 
        weightVec = np.array([ 1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,
  1, -1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1 ,1  ,1  ,1  ,1  ,1 ,-1, 1, 1], dtype = np.int8)
        weightVec = np.concatenate((weightVec, fillerHalfNeg))
        weightMatrix = np.zeros((numEpochs, len(mfs)), dtype = np.int8)
        weightMatrix[0] = weightVec

        for epoch in range(numEpochs - 1):
            # Iterate through all the epochs 
            print('************************* Epoc ',epoch, '********************************')
            #print(weightMatrix)
            #count = 0 #Counter to keep track of iterations 
            error_i = 10000000000
            # The original weight before training 
            oldWeight = weightMatrix[epoch]

            for i in range(len(allDataTrain)):

                #Quantum product 
                psi_i = np.concatenate((allDataTrain[i], fillerAllNeg)) #Adding filler 
                #print('mfs == mfsClassic: ', weightVec == mfsClassic)
                proj_i = quantumProd(qc, weightMatrix[epoch], psi_i, nshots) #Perform quantum projection 

                predictions_train[i] = 1*(proj_i >= treshold) 

                error_i = classificationTrain[i] - predictions_train[i]
                
                #Learning rate
                
                for n in range(len(mfs)):
                    #mfs[n] = mfs[n] - 2*mfs[n]*error_i**2
                    # Take the input as reference, whenever they differ from the weights, change the value 
                    isEqual = np.int8(psi_i[n] == weightMatrix[epoch][n])

                    flip = np.int8((not(classificationTrain[i]) and predictions_train[i] and isEqual) or (classificationTrain[i] and not(predictions_train[i]) and not(isEqual)))
                    #print(flip)
                    # Now the learning rate will be applied on the elements that the weight and input vector concide or not depending if is a false positive or a flase negative, with a random layer. 
                    #mfs[n] = mfs[n]
                    weightMatrix[epoch + 1][n] = oldWeight[n]*((-1)**(flip * np.random.randint(0,2,dtype=np.int8)))
                    #weightMatrix[epoch + 1][n] = oldWeight[n]*((-1)**(flip))
                    #weightMatrix[epoch + 1][n] = ((-1)**(np.random.randint(0,2)))

                    
                    # mfs should be updated 
                    #mfs[n] = mfs[n] - (flip)*2*mfs[n]*error_i**2
            # Now the weight veector should be updated 
            print('Just updated the mfs: ', weightMatrix[epoch + 1])
            print('Is it equal? ', (weightMatrix[epoch] == weightMatrix[epoch + 1]).all())
            
            #Validation of the epoch    
            for i in range(len(allDataVal)):
                # Now we validate this updated weight
                #Quantum 
                psi_i = np.concatenate((allDataVal[i], fillerAllNeg))

                #proj_i = (1/norm)*np.dot(allDataVal[i], mfs)
                proj_i = quantumProd(qc, weightMatrix[epoch], psi_i, nshots)

                #print(1*(proj_i >= treshold))
                #predictions_val = np.append(predictions_val, np.int8(1*(proj_i >= treshold)))
                predictions_val[i] = 1*(proj_i >= treshold)

                #error_i = classificationVal[i] - predictions_val[i]
                
                #Classical case  
                clsProj_i = weightMatrix[epoch]@psi_i/(len(mfs))
                #print(mfs)
                #print(psi_i)
                #print((proj_i , clsProj_i))
                clsPredictions_val[i] = 1*(clsProj_i >= treshold)
                #clsPredictions_val = np.append(clsPredictions_val, 1 * (clsProj_i >= treshold))
            
                #Learning rate
            
            #print(clsPredictions_val)
            #print(predictions_val)
            # Quantum ConfusionMatrix
            cfMatVal = confusion_matrix(classificationVal, predictions_val)
            efficiency = accuracy_score(classificationVal, predictions_val)*100
            print('Quant eff', efficiency)
            #efficiencyArr[epoch] = efficiency
            efficiencyArr = np.append(efficiencyArr, efficiency)
            
            #Classic ConfusionMatrix
            clsCfMatVal = confusion_matrix(classificationVal, clsPredictions_val)
            clsEfficiency = accuracy_score(classificationVal, clsPredictions_val)*100
            #print('Classixc eff: ', clsEfficiency)
            #clsEfficiencyArr[epoch] = clsEfficiency
            clsEfficiencyArr = np.append(clsEfficiencyArr, clsEfficiency)
            #print(clsPredictions_val == predictions_val)
            #print(predictions_val)
            #print(clsPredictions_val)
            thQ = len(np.where(predictions_val == 0))
            thC = len(np.where(clsPredictions_val == 0))
            print('Does the th on quantum and classical are the same?', thQ == thC)
            print('Should be the efficieny in both cases the same?', thQ == thC)
    
            #print((efficiency, clsEfficiency))
            #print(efficiencyArr)
            print('Quantum and classic efficiencies: ', (efficiency, clsEfficiency))
            #mse error of the epoch
            #print('Modded weight for the epoch', epoch,':', mfs)
            
            #weightHistoryVec[epoch] = mfs 
            #weightHistoryVec[epoch] = np.delete(mfs, np.arange(numBits*numBars,len(mfs)))
            #print('efficiency of ', epoch, ':', efficiencyArr[epoch])
            #prin('quantum efficieny of', epoch, ':', clsEfficiency[epoch])
            #print('This epoch had', str(count), 'projections', 'with mse=', mseVect[epoch])

            '''
            TODO: 
            Change the treshold and calculate efficiency, confussion matrix

            '''

            


        print('Trained:', mfs)
        '''
        plt.plot(np.arange(len(efficiencyArr)), efficiencyArr)
        plt.savefig('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/efficiencyRandom_Assisted_'+str(numEpochs)+'Epochs_treshold_'+str(treshold)+'_numBits'+str(numBits)+'.png',bbox_inches='tight', dpi=199)
        '''
        minErrorIndex = np.argmax(efficiencyArr)
        print(efficiencyArr)
        minErrorWeigth = weightMatrix[minErrorIndex]
        



        '''
                
        ▀█▀ █▀▀ █▀ ▀█▀   █▀ █▀▀ █▀▀ ▀█▀ █ █▀█ █▄░█
        ░█░ ██▄ ▄█ ░█░   ▄█ ██▄ █▄▄ ░█░ █ █▄█ █░▀█
        
        █▀▄▀█ █ █▄░█ █ █▀▄▀█ █░█ █▀▄▀█   █▀▀ █▀█ █▀█ █▀█ █▀█
        █░▀░█ █ █░▀█ █ █░▀░█ █▄█ █░▀░█   ██▄ █▀▄ █▀▄ █▄█ █▀▄
        
        '''

        
        for i in range(len(allDataTest)):

            #proj_i = (1/norm)*np.dot(allDataTest[i], minErrorWeigth)
            #predictions_test[i] = 1*(proj_i >= treshold)
            #mfs[n] = mfs[n] - (flip * np.random.randint(2))*2*mfs[n]*error_i**2.1
            #proj_i = (1/norm)*np.dot(allDataTest[i], minErrorWeigth)
            psi_i = np.concatenate((allDataTest[i], fillerAllNeg))
            proj_i = quantumProd(qc, minErrorWeigth, psi_i, nshots)
            #print(proj_i)
            predictions_test[i] = 1*(proj_i >= treshold)

            # Classical Case
            clsProj_i = minErrorWeigth@psi_i
            clsPredictions_test[i] = 1*(clsProj_i >= treshold)

        qECf = confusion_matrix(classificationTest, predictions_test)
        qEfficiency = (np.trace(qECf) / len(predictions_test)) * 100

        #Classical Case 
        clsCfTest = confusion_matrix(classificationTest, clsPredictions_test)
        clsEfficiency = (np.trace(clsCfTest) / len(clsPredictions_test))*100
        
        # Plotting the max efficiency ConfusionMatrix
        plt.close()
        qECfDisp = ConfusionMatrixDisplay(qECf)
        qECfDisp.plot()
        plt.title('Quantum Perceptron QEff(Test): '+str(qEfficiency))
        plt.savefig('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/TEST-QuantumPerceptron_'+str((train_percentage - 1)/2)+'_numBits'+str(numBits)+str(treshold)+'_treshold_'+str(nshots)+'_nshots.png', dpi=199)
        plt.close()

        clsECfDisp = ConfusionMatrixDisplay(clsCfTest)
        clsECfDisp.plot()
        plt.title('Classical Perceptron Eff: '+str(clsEfficiency))
        plt.savefig('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/TEST-ClassicalPerceptron_'+str((train_percentage - 1)/2)+'_numBits'+str(numBits)+str(treshold)+'_treshold_'+str(nshots)+'_nshots.png', dpi=199)
        plt.close()





        # Saving the weight vectors 
        np.savetxt('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/minerrorWeight.csv',minErrorWeigth)
        np.savetxt('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/weigthVectors/weightMat.csv',weightMatrix)
        np.savetxt('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/efficiency/eff_' + str(numEpochs) + '_epochs_'+str(treshold)+'_treshold'+'_numBits_'+str(numBits)+'_.csv', efficiencyArr)

        #efficiencyTreshold.append(efficiencyArr[minErrorIndex])
        print(efficiencyArr)
        print(clsEfficiencyArr)
        print(efficiencyArr == clsEfficiencyArr)
        plt.close()
        plt.plot(np.arange(len(efficiencyArr)), efficiencyArr, color = 'g', label = 'quantum')
        plt.plot(np.arange(len(efficiencyArr)), clsEfficiencyArr, color = 'r', label = 'classic')
        plt.xlabel('Epoch')
        plt.ylabel('Efficiency')
        plt.title('Quantum and Classical Efficiencies')
        plt.legend()
        plt.savefig('/home/guerrero/Documents/UNAM/QuantumComputing/PerceprontV1/transition/14-07-24/Efficencies_'+str(tresholdRange[0])+'-'+ str(tresholdRange[-1])+'--'+ str(numEpochs) + '_epocs'+'_numBits'+str(numBits)+'.png', bbox_inches ='tight',dpi=199)

        #print('Is the most efficient weightVec the same as the classic case? '+ str(minErrorWeigth == mfsClassic))
        print(mfsClassic)
        print(minErrorWeigth)
#'''
def main():
    #testProjections()
    train()
    




if __name__ == main():
    main()
#'''
