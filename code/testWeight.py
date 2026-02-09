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




def plotHist(projCoh, projTh, treshold, diff, assisted): 
    projCoh = np.array(projCoh)
    projTh = np.array(projTh)
    _, edges = np.histogram(projCoh, 'auto')
    n_bins = len(edges) 
    #thHist = np.histogram(projTh, n_bins)
    mix = np.concatenate((projCoh, projTh))
    histRange = (np.min(mix), np.max(mix))
    plt.hist(x = projCoh,bins = n_bins, alpha = 0.75, color = '#0043DA', range =histRange )
    plt.hist(x = projTh, bins =  n_bins, alpha = 0.75, color = '#DBCB00', range = histRange)
    plt.axvline(x = treshold, color = 'r', label = 'treshold', linestyle = '--')
    plt.legend()
    plt.title(str(diff)  + ' treshold: ' + str(treshold))
    plt.savefig(str(diff) +'_' + str(assisted) + 'testWeightTreshold_' + str(treshold) + '.png', bbox_inches = 'tight')
    plt.close()

def testRealSession(train, w, treshold):
    # ***************** Get Data *********************
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data Preparation **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(np.ceil(llData*testPerc))
    print('train:', train)
    print('test:', test)


    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)


    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    #********************** Quantum Circuits ******************** 
    nshots = 2**13
    #nshots = int(input('shots'))

    circuitsCoh = qml.getProjCircuits(7, coh, w)
    circuitsTh = qml.getProjCircuits(7, th, w)
    # ********************** Execution ************************
    job_qrealCoh = tr.realExecuteSessionMulti(circuitsCoh, nshots)
    job_qrealTh = tr.realExecuteSessionMulti(circuitsTh, nshots)
    
    realCoh = tr.getResultsFromJobs(job_qrealCoh)
    realTh = tr.getResultsFromJobs(job_qrealTh)

    print('finished Session')



    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    
    mix = np.concatenate((simuCoh, simuTh))   
    classification = np.zeros(len(mix)) 
    classification[:len(simuCoh)] = 1

    
    predictions = np.where(mix >= treshold, 1, 0)

    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))

    plotHist(simuCoh, simuTh, treshold, 'qiskit')
    

    return accuracy, mix



def getProjBatch(train, w, treshold):
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(np.ceil(allData*testPerc))
    print('train:', train)
    print('test:', test)


    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)


    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    # Generate circuits 
    nshots = 2**13
    #nshots = int(input('shots'))

    circuitsCoh = qml.getProjCircuits(7, coh, w)
    circuitsTh = qml.getProjCircuits(7, th, w)
    
    pubCoh = tr.realExecuteBatch(circuitsCoh, nshots)
    pubTh = tr.realExecuteBatch(circuitsTh, nshots)
    
    qrealCoh = tr.getResultsBatch(pubCoh)
    qrealTh = tr.getResultsBatch(pubTh)
    np.savetxt('cohBatch.csv', qrealCoh, delimiter = ',')
    np.savetxt('ThBatch.csv', qrealTh, delimiter = ',')
    print('finished aer simulation')

    mix = np.concatenate((qrealCoh, qrealTh))   
    classification = np.zeros(len(mix)) 
    classification[:len(simuCoh)] = 1
    return mix, classification

    
    predictions = np.where(mix >= treshold, 1, 0)

    print(np.sum(predictions))
    np.savetxt('testWeightPred.csv', predictions, delimiter = ',')
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))

    plotHist(simuCoh, simuTh, treshold, 'qiskit')
    

    return accuracy, mix

def testRealBatch(projections, treshold, identifier, assisted):
    
    #Step functio
    projections = np.array(projections)
    predictions = np.where(projections <= treshold, 1, 0)
    
    #Constructing original dataset
    cohlen = int(len(projections)/2)
    classification = np.zeros_like(projections)
    classification[:cohlen] = 1
    batchCoh = projections[:cohlen]
    batchTh = projections[cohlen:]
    
    #Accuracy
    accuracy = accuracy_score(classification, predictions)
    print('Real QPU accuracy: ', accuracy)
    np.savetxt('RealQPU_Projections.csv', projections)
    np.savetxt('RealQPU_Acc.txt', np.array([accuracy]))
    plotHist(batchCoh, batchTh, treshold, 'Batch', str(assisted))
    return accuracy, projections




def testClassic(train, w, treshold, assisted):
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    
    allData =  int(train / trainPerc)
    test = int(np.ceil(allData*testPerc))
    print('subset:',allData)
    print('train:', train)
    print('test:', test)



    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    #circuitsCoh = qml.getProjCircuits(7, coh, w)
    #circuitsTh = qml.getProjCircuits(7, th, w)
    #print(circuits)
    #c = np.dot(i,w)**2 
    #p = tr.simulate(qml.getProjCircuits(coh,w), nshots) * 64

    #ran = np.arange(4,21,1)
    #ran = np.arange(1, 3, 1)
    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    for i in range(len(coh[0])):
        clasCoh.append(np.abs(coh[:,i]@w / l)**2)
        clasTh.append(np.abs(th[:,i]@w / l)**2)

    mix = np.concatenate((clasCoh, clasTh))   
    classification = np.zeros(len(mix))
    classification[:len(clasCoh)] = 1


    predictions = np.where(mix >= treshold, 1, 0)
    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))
    plotHist(clasCoh, clasTh, treshold, 'classic', assisted)

    return accuracy, mix




def testSeqQiskit(train, w, treshold, assisted):
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(np.ceil(allData*testPerc))
    print('train:', train)
    print('test:', test)



    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)

    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    nshots = 2**13
    for i in range(len(coh[0])):
        
        clasCoh.append(qml.singleProjAer(coh[:,i], w, 7, nshots))
        clasTh.append(qml.singleProjAer(th[:,i], w, 7, nshots))
        

    mix = np.concatenate((clasCoh, clasTh))   
    classification = np.zeros(len(mix))
    classification[:len(clasCoh)] = 1


    predictions = np.where(mix >= treshold, 1, 0)
    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))
    plotHist(clasCoh, clasTh, treshold, 'ideal', assisted)

    return accuracy, mix


def testFake(train, w, treshold, backend, assisted):
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(np.ceil(allData*testPerc))
    print('train:', train)
    print('test:', test)



    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)

    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    nshots = 2**13
    for i in range(len(coh[0])):
        
        clasCoh.append(qml.singleProjFake(coh[:,i], w, 7, nshots, backend))
        clasTh.append(qml.singleProjFake(th[:,i], w, 7, nshots, backend))
        

    mix = np.concatenate((clasCoh, clasTh))   
    classification = np.zeros(len(mix))
    classification[:len(clasCoh)] = 1


    predictions = np.where(mix >= treshold, 1, 0)
    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    #print(np.nonzero(predictions))
    plotHist(clasCoh, clasTh, treshold, 'noisy', assisted)

    return accuracy, mix



def saveIBM():

    QiskitRuntimeService.save_account(
    token='4a4d0684ed263c22a0dc741a2f75ad4ea114dbb77f9aa41be512ccadfd36c8f4aa7d2a408a4b02fff74dee5493e694d29f937351cf482d1b662824110131a74d',
    channel="ibm_quantum" # `channel` distinguishes between different account types
    )

def typeTest(hint, thermal, coherent, text):
    # I already did the projections, that's why im asking for them
    # So I have to sum over all the projections
    # varying the limits

    # 1) Create the histograms
    
    mix = np.concatenate((thermal, coherent))
    histRange = (np.min(mix), np.max(mix)) # Only to get the adequate range 
    
    # Histograms 
    cohHist, cohEdges = np.histogram(a = coherent, bins = 'auto', range = histRange)
    n_bins = len(cohEdges)
    thHist, thEdges = np.histogram(a = thermal, bins = n_bins, range = histRange)
    
    
    
    
    cohSum = cohHist.sum()
    thSum = thHist.sum()
    histlen = len(cohHist)
    
    ran = np.arange(0,histlen)
    
    # 2) initialize probabilities 
    p_alpha = np.zeros(histlen)
    p_beta = np.zeros(histlen)

    # 3) Get the P(alpha) and P(beta): Loop over the histogram, note that we do not take into account the edges
    
    for i in ran:

        coh_pa = np.sum(cohHist[i + 1:]) 
        th_pa = np.sum(thHist[i + 1:]) 

        p_alpha[i] = (coh_pa + th_pa) / (cohSum + thSum)

        coh_pb = np.sum(cohHist[:i + 1]) 
        th_pb = np.sum(thHist[:i + 1]) 

        p_beta[i] = (coh_pb + th_pb) / (cohSum + thSum)

    # 3) Now create a new edges array with the midpoint of two edges 


    midPointEdges = np.zeros(n_bins - 1)
    
    for i in range(n_bins - 1):
        midPointEdges[i] = (cohEdges[i] + cohEdges[i + 1]) / 2
    
    # 4) Calculate the differences between the probabilities 
    diff = np.abs(p_alpha - p_beta)
    trsholdId = np.argmin(diff)

    # 5) Use the id to get the treshold 
    treshold = midPointEdges[trsholdId]
    print('min diff: ', diff[trsholdId])
    print('Between edges: ', cohEdges[trsholdId], ',', cohEdges[trsholdId + 1] )
    print('p(a), p(b) : ', p_alpha[trsholdId], ',', p_beta[trsholdId])
    plt.plot(midPointEdges, p_alpha, color = 'r', label = 'Error Type I')
    plt.plot(midPointEdges, p_beta, color = 'b', label = 'Error Type II')
    plt.ylabel('Probability')
    plt.xlabel('Tresholds')
    plt.title('Error Test Type I and Type II')
    plt.legend()
    plt.savefig(fname = str(text) + '_typeTestError.pdf', format = 'pdf', pad_inches =  'layout', dpi = 'figure')
    plt.close()
    print('Treshold: ', treshold)
    return treshold

def plotAccs(counts, text):

    # Assuming acc, accQ, and accF are defined
    fig, ax = plt.subplots()

    accs = ['Classic', 'Qiskit Aer', 'Qiskit Fake B.']
    #counts = [acc, accQ, accF]
    
    bar_colors = ['#F07800', '#0081F0', '#214B70']
    

    bars = ax.bar(accs, counts, color=bar_colors)

    # Add value annotations on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X coordinate
            height,                             # Y coordinate (top of the bar)
            f'{height:.2f}',                    # Display value with 2 decimal places
            ha='center',                        # Horizontal alignment
            va='bottom'                         # Vertical alignment
        )

    ax.set_ylabel('Accuracy')
    ax.set_title('Type')
    ax.legend(title='Comparative of perceptrons')

    plt.savefig(str(text) + '_AccuracyCQF.png', bbox_inches='tight')



def globalTest(w, subs, treshold, text, backend):
    import pandas as pd 
    print('classic test')
    clsAcc,_ = testClassic(subs, w, treshold, text)
    print('qiskit test')
    qiskAcc,_ = testSeqQiskit(subs, w, treshold, text)
    print('fake test')
    fakeAcc,_ = testFake(subs, w, treshold, backend, text)

    header = np.array(['classic', 'qiskit', 'fake'])

    data = pd.DataFrame((clsAcc, qiskAcc, fakeAcc))
    data.to_csv(str(text) + 'GlobalTestResults.csv')
    
    plotAccs([clsAcc, qiskAcc, fakeAcc], text)
    return clsAcc, qiskAcc, fakeAcc
    
    
def main():
    saveIBM()
    #w = [1,-1,-1,1,1,1,1,1,1,-1,1,-1, 1,-1,1,-1,1,1, 1,1,1,-1, 1,-1,1,1, 1,1,1,-1, 1,-1,1,1, 1,1, 1,1,1,1, 1,1,1,1, -1, 1]
    #w = [1, -1,-1,-1, -1,1,-1,1,  1,-1,-1,1,   -1,-1,1,-1,  1,1, -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,1,1,  1,1,  1,1,1,-1,  1,1,-1,1,  1,1,1,1,  1,1,1,1,  1,1, 1,1,1,1,1,1,1,1,1,1]

    w = pd.read_csv('09115bestSingleCh.csv', delimiter = ',', header = None)
    print(len(w))

    
    accQ, mixQiskit = test(1000, w, 0.24853515625)
    acc, mixClass = testClassic(1000, w, 0.24853515625)
    dif = []
    for i in range(len(mixQiskit)):
        dif.append(np.abs(mixQiskit[i] - mixClass[i]))


    plt.plot(mixQiskit, "-r", label = 'Qiskit')
    plt.plot(mixClass, "-b", label = 'Classic')
    plt.plot(dif, "-g", label = 'Abs Difference')
    plt.legend(loc="upper left")
    plt.title('Perceptrón cuántico (Qiskit) vs clásico')
    #plt.show()
    plt.savefig('testWeightCuanticoVsClasico.png', bbox_inches = 'tight')
    plt.close()

    print('Fake parallel: ', accQ)
    print('Classic: ', acc)


    #accReal, mixReal = testRealSession(100, w, 0.24853515625)
    #np.save_txt('accRealSession.csv', accReal, delimeter = ',')
    #print('Q Real: ', accReal)


#main()
