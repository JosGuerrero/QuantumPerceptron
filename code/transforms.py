from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile, assemble
import qiskit_ibm_runtime.fake_provider as fak
from qiskit_ibm_runtime.fake_provider import FakeBrisbane 
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
import bitarray as bta
from qiskit.visualization import circuit_drawer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Session
import transforms as tr
import concurrent.futures
from typing import List
# Ui
def u1(qc):
    qc.cz(0,1)
def u2(qc):
    qc.z(1)
    u1(qc)
def u3(qc):
    qc.z(1)
def u4(qc):
    qc.z(0)
    qc.cz(0,1)
def u5(qc):
    qc.z(0)
def u6(qc):
    qc.z([1,0])
def u7(qc):
    u6(qc)
    u1(qc)
def u8(qc):
    u1(qc)
    qc.x([0,1])
def u9(qc):
    qc.cz(0,1)
    qc.x([0,1])
    qc.cz(0,1)
def u10(qc):
    u5(qc)
    qc.x([0,1])
def u11(qc):
    qc.cz(0,1)
    qc.x([0,1])
    qc.z(1)
def u12(qc):
    u3(qc)
    qc.x([0,1])
def u13(qc):
    qc.cz(0,1)
    qc.x([0,1])
    qc.z(0)
def u14(qc):
    u7(qc)
    qc.x([0,1])
def u15(qc):
    qc.z(1)
    qc.x([0,1])
    qc.z(1)


# Uw
def uw_1(qc):
    qc.cz(0,1)
def uw_2(qc):
    u1(qc)
    qc.z(1)

def uw_3(qc):
    qc.z(1)
def uw_4(qc):
    qc.cz(0,1)
    qc.z(0)

def uw_5(qc):
    qc.z(0)

def uw_6(qc):
    qc.z([1,0])
def uw_7(qc):
    u1(qc)
    u6(qc)

def uw_8(qc):
    qc.x([0,1])
    u1(qc)

def uw_9(qc):
    qc.cz(0,1)
    qc.x([0,1])
    qc.cz(0,1)

def uw_10(qc):
    qc.x([0,1])
    u5(qc)

def uw_11(qc):
    qc.z(1)
    qc.x([0,1])
    qc.cz(0,1)

def uw_12(qc):
    qc.x([0,1])
    u3(qc)

def uw_13(qc):
    qc.z(0)
    qc.x([0,1])
    qc.cz(0,1)

def uw_14(qc):
    qc.x([0,1])
    u7(qc)

def uw_15(qc):
    qc.z(1)
    qc.x([0,1])
    qc.z(1)

def discrepancy(qprob, cprob):
    rows, cols = np.shape(qprob)
    disMat = np.zeros_like(qprob)
    for i in range(rows):
        for j in range(cols):
            disMat[i,j] = np.abs(qprob[i,j] - cprob[i,j])
    return disMat
def avg_discrepancy(dismat):
    avg = 0
    rows, cols = np.shape(dismat)
    for i in range(rows):
        for j in range(cols):
            avg += dismat[i,j]/(rows*cols)
    return avg

def zero2minus(zerosArray):
    # Create a byte 
    minusArray = (-1)*(np.ones(len(zerosArray), dtype = np.int8))
    for i in range(len(zerosArray)):
        minusArray[i] = minusArray[i]**(np.int8(zerosArray[i]))
    return minusArray

def getUnitary(ones_vec):
    #Note use fliplr when ki > ki/2
    
    #ones_vec = zero2minus(bin_vector)
    print(ones_vec)
    #ones_vec = np.concatenate((ones_vec,ones_vec))
    #print(ones_vec)
    dim = len(ones_vec)
    unit = np.zeros((dim,dim), dtype = np.int8)
    #print(np.shape(unit))
    #print(dim)
    #print(unit)

    for i in range(dim):
            unit[i][i] = ones_vec[i] 
    return unit


def ui(qc, inputVec): 
    N = qc.num_qubits
    print(N)
    rows = 2**(N)
    target = np.arange(N).tolist()
    ancilla = 2**1
    
    #inputVec = bta.bitarray(np.binary_repr(ki, 2**(N - 1)))
    uiMat = getUnitary(inputVec)
    #print(uiMat)
    #print(np.shape(uiMat))
    

    uiMatAnc = np.kron(np.eye(ancilla, dtype = np.int8), uiMat)
    #print(np.shape(uiMatAnc))

    #print(uiMatAnc)
    Ui = UnitaryGate(uiMatAnc)
    #print(target)
    qc.append(Ui, target)

    print('inputVec:', inputVec)
    print('StateVec:', Statevector(qc))

def ui_rev(qc, inputVec): 
    N = qc.num_qubits
    print(N)
    rows = 2**(N)
    target = np.arange(N).tolist()
    ancilla = 2**1
    
    #inputVec = bta.bitarray(np.binary_repr(ki, 2**(N - 1)))
    uiMat = np.fliplr(getUnitary(inputVec))

    uiMatAnc = np.kron(np.eye(ancilla, dtype = np.int8), uiMat)

    print(uiMatAnc)
    Ui = UnitaryGate(uiMatAnc)
    print(target)
    qc.append(Ui, target)

    

def uw(qc, weigthVec): 
    N = qc.num_qubits
    print(N)
    rows = 2**(N)
    target = np.arange(N).tolist()
    ancilla = 2**1
    
    #weigthVec = bta.bitarray(np.binary_repr(kw, 2**(N - 1)))
    uwMat = getUnitary(weigthVec)

    uwMatAnc = np.kron(np.eye(ancilla, dtype = np.int8), uwMat)

    #print(uwMatAnc)
    Uw = UnitaryGate(uwMatAnc)
    #print(target)
    qc.append(Uw, target)

def uw_rev(qc, weigthVec): 
    N = qc.num_qubits
    print(N)
    rows = 2**(N)
    target = np.arange(N).tolist()
    ancilla = 2**1
    
    #weigthVec = bta.bitarray(np.binary_repr(kw, 2**(N - 1)))
    uwMat = np.fliplr(getUnitary(weigthVec))

    uwMatAnc = np.kron(np.eye(ancilla, dtype = np.int8), uwMat)

    print(uwMatAnc)
    Uw = UnitaryGate(uwMatAnc)
    print(target)
    qc.append(Uw, target)


def simulate_sv(qc, n_shots): 
   
    # 1) Create Backend 

    sim = AerSimulator(method = 'statevector')

    # 1.1) Transpile 
    qc = transpile(qc, sim)

    # 2) Sampler V2 instance 
    sampler = SamplerV2(sim)

    # 3) Run the sampler 
    job = sampler.run(qc, shots = n_shots)
    
    # 4) Retrive results 
    results_sim  = job.result()

    return results_sim

def getResultsMultiCirc(simResults):
    pub_result = simResults[0]
    countList = []
    nshots = pub_result.data.c.num_shots
    counts = pub_result.data.c.get_counts()        #Si una probabilidad es 0
    counts.setdefault('00', 0)
    counts.setdefault('01', 0)
    
    return counts

def getResultsAerBackend(simResult):
    pub_result = simResult[0]
    #get counts 
    countsList = []
    #classicalReg = pub_result.c
    #for cbit in classicalReg:
    
    nshots = pub_result.data.c.num_shots
    counts = pub_result.data.c.get_counts()        #Si una probabilidad es 0    
    counts.setdefault('0', 0)
    counts.setdefault('1', 0)
    print(counts)
    result = counts['1'] / nshots
    return result



def simulateMultiSCirc(qc,n_shots, backend): 
      
    '''
    # The backend needs to be outside the excecution loop
    # get a real backend from the runtime service

    service = QiskitRuntimeService()    
    backend = service.least_busy(operational = True, simulator = False, min_num_qubits = qc.num_qubits)
    backend = service.backend("ibm_brisbane")
    '''
    # generate a simulator that mimics the real quantum system with the latest calibration results
    backend_sim = AerSimulator.from_backend(backend)
    backend_sim.set_options(device="GPU") 
    #1) Define Fake Backend 
    
    # Pre transpilation

    # 2) Create the pass manager 
    pm = generate_preset_pass_manager(optimization_level=3, backend = backend_sim)
    
    
    # 4) Transform into an ISA circuit (Transpilation)
    isaCircuit = pm.run(qc)
    #print('isaCircuit:')
    #circuit_drawer(isaCircuit[0], output="text", style={"backgroundcolor": "#EEEEEE"})
    
    # New> Number of gates>
    print('num of gates: ', len(isaCircuit[0].data))

    # 5) Create the sampler object 
    sampler = SamplerV2(backend_sim)
    
    # 6) Create the job 
    simjob = sampler.run(isaCircuit, shots = n_shots)
    simResult = simjob.result()
    #print(np.shape(job.result()))
    #simResult = tr.getResultsFromJobs(simjob)
    print(simResult)
    result = getResultsMultiCirc(simResult) 

    return result



def simulateAerBackend(qc,n_shots, backend): 
      
    '''
    # The backend needs to be outside the excecution loop
    # get a real backend from the runtime service

    service = QiskitRuntimeService()    
    backend = service.least_busy(operational = True, simulator = False, min_num_qubits = qc.num_qubits)
    backend = service.backend("ibm_brisbane")
    '''
    # generate a simulator that mimics the real quantum system with the latest calibration results
    backend_sim = AerSimulator.from_backend(backend)
    backend_sim.set_options(device="GPU") 
    #1) Define Fake Backend 
    
    # Pre transpilation

    # 2) Create the pass manager 
    pm = generate_preset_pass_manager(optimization_level=3, backend = backend_sim)
    
    
    # 4) Transform into an ISA circuit (Transpilation)
    
    isaCircuit = pm.run(qc)
    #print('isaCircuit:')
    #circuit_drawer(isaCircuit[0], output="text", style={"backgroundcolor": "#EEEEEE"})
    
    print('num of gates: ', len(isaCircuit[0].data))
    # 5) Create the sampler object 
    sampler = SamplerV2(backend_sim)
    
    # 6) Create the job 
    simjob = sampler.run(isaCircuit, shots = n_shots)
    simResult = simjob.result()
    #print(np.shape(job.result()))
    #simResult = tr.getResultsFromJobs(simjob)
    print(simResult)
    result = getResultsAerBackend(simResult)
    
    return result



def simulate(qc, n_shots): 
   
    
    #1) Define Fake Backend 

    backend = FakeBrisbane()
    # Pre transpilation

    # 2) Create the pass manager 
    pm = generate_preset_pass_manager(optimization_level=0, backend = backend)
    #pm = generate_preset_pass_manager(optimization_level = 0)

    # 4) Transform into an ISA circuit (Transpilation)
    print('Transpiling')
    isaCircuit = pm.run(qc)
    print(isaCircuit)

    # 5) Create the sampler object 
    sampler = SamplerV2(backend)
    
    # 6) Create the job 
    job = sampler.run(isaCircuit, shots = n_shots)
    #print(np.shape(job.result()))
    
    # 7) Return the job's result 
    return job.result()
    
def getResultsFromJobs(jobResults):
    # Once we got the job results, we got to retrive them 
    #print(jobResults[0])
    #rows = len(jobResults)
    results = []

    for result in jobResults:
        #i = 1
        #print(i)
        #print(result)
        #i+=1

        pub_result = result
        #get counts 
        nshots = pub_result.data.c.num_shots
        print('nshots: ',nshots)

        counts = pub_result.data.c.get_counts()        #Si una probabilidad es 0
        print(counts)
        counts.setdefault('0', 0)
        counts.setdefault('1', 0)
        print(counts)
        results.append(counts['1'] / nshots)

    return results

def saveAccount(myToken, ovwrt):
    QiskitRuntimeService.save_account(
            channel = "ibm_quantum",
            token = myToken,
            set_as_default = True,
            overwrite = ovwrt
            )


def realExecute(qc):
    #print(qc)
    numQubits = qc.num_qubits
    # We got a define a runtime service 
    #print(qc)
    numQubits = qc.num_qubits
    # We got a define a runtime service 
    # We got a define a runtime service 
    numQubits = qc.num_qubits
    # We got a define a runtime service 
    #print('Saved accounts=', QiskitRuntimeService.saved_accounts())

    # 1) Start service 
    service = QiskitRuntimeService()
    
    #2) Define Backend 
    backend = service.least_busy(operational = True, simulator = False, min_num_qubits = numQubits)
    
    #Pre steps to transpilation

    # 3) Create the pass manager 

    pm = generate_preset_pass_manager(optimization_level=0, backend = backend)
    #pm = generate_preset_pass_manager(optimization_level = 0)

    # 4) Transform into an ISA circuit 
    print('Transpiling')
    isaCircuit = pm.run(qc)
    #print('Qubits of transpiled circuit:', isaCircuit.num_qubits)

    # 5) Create the Sampler 
    sampler = SamplerV2(backend)

    # 6) Create the job 
    job = sampler.run([isaCircuit])
    
    print(f"-> Estimated qtime:{job.usage_estimation}")
    print(f"-> Job ID:{job.job_id()}")
    print(f"-> Job Satus: {job.status()}")
    result= job.result()
    
    # 7) Check results 
    pub_result = result[0]
    #get shots 
    nshots = pub_result.data.c.num_shots
    #get counts 
    counts = pub_result.data.c.get_counts()        #Si una probabilidad es 0
    counts.setdefault('0', 0)
    counts.setdefault('1', 0)
    print(counts)
    # 8) Return the results 
    return counts['1']/nshots




def realExecuteSession(qc):
    #print(qc)
    numQubits = qc.num_qubits
    # We got a define a runtime service 
    #print('Saved accounts=', QiskitRuntimeService.saved_accounts())
    service = QiskitRuntimeService()
    #By the time, this qp is not accepting jobs 
    #backend = service.least_busy(operational = True, simulator = False, min_num_qubits = numQubits)
    
    backends = service.backends()
    backend = backends[0]
    #Pre steps to transpilation
    pm = generate_preset_pass_manager(target = backend.target, optimization_level=1)
    passedCirquit = pm.run(qc)
    #print('Passed circuit')
    #print(passedCirquit)
    with Session(service = service, backend=backend, max_time="25m") as session:
        # Submit a session  
        sampler = SamplerV2(session = session)
        job = sampler.run([passedCirquit])
        print(f"-> Job ID:{job.job_id()}")
        print(f"-> Job Satus: {job.status()}")

        #print('Estimated time (quantum seconds):', job.usage_estimation)
        result= job.result()
        pub_result = result[0]
        #get shots 
        nshots = pub_result.data.c.num_shots
        #get counts 
        counts = pub_result.data.c.get_counts()        #Si una probabilidad es 0
        counts.setdefault('0', 0)
        counts.setdefault('1', 0)
        print(counts)
        
        return counts['1']/nshots

def realExecuteSessionMulti(qc, nshots):

    #print(qc)
    numQubits = qc[0].num_qubits
    # We got a define a runtime service 
    #print('Saved accounts=', QiskitRuntimeService.saved_accounts())
    service = QiskitRuntimeService()
    #By the time, this qp is not accepting jobs 
    backend = service.least_busy(operational = True, simulator = False, min_num_qubits = numQubits)


    #Pre steps to transpilation
    pm = generate_preset_pass_manager(target = backend.target, optimization_level=1)
    passedCirquit = pm.run(qc)
    #print('Passed circuit')
    #print(passedCirquit)
    with Session(service = service, backend=backend) as session:

        # Submit a session  
        sampler = SamplerV2(session = session)
        sampler.options.max_execution_time = 12500
        job = sampler.run(passedCirquit, shots = nshots)
        print(f"-> Job ID:{job.job_id()}")
        print(f"-> Job Satus: {job.status()}")

        print('Estimated time (quantum seconds):', job.usage_estimation)
        result= job.result()
        return result 
       
       
       
def realExecuteBatch(circuits, nshots, backend):
    pm = generate_preset_pass_manager(backend = backend, optimization_level = 1)
    isa_circuits = pm.run(circuits)


    max_circuits = 250 
    all_partitioned_circuits = []
    for i in range(0, len(isa_circuits), max_circuits):
        all_partitioned_circuits.append(isa_circuits[i : i + max_circuits])
    jobs = []
    jobs_pub_results = []
    counts = []
    start_idx = 0

    with Batch(backend=backend, max_time = "25m"):
        sampler = SamplerV2()
        sampler.options.max_execution_time = 12500
        for partitioned_circuit in all_partitioned_circuits:
            job = sampler.run(partitioned_circuit, shots = nshots)
            jobs.append(job)
            job_pub = job.result()[0]
            jobs_pub_results.append(job_pub)
    return jobs_pub_results 

def sendBatch(circuits, nshots, backend):
    jobs = [] # I need a job per each circuit
    job_pubs = []
    pm = generate_preset_pass_manager(backend = backend, optimization_level = 1)
    isa_circuits = pm.run(circuits)
    job_id = 'no_job_id'
    with Batch(backend = backend) as batch:
        sampler = SamplerV2()  
        simjob = sampler.run(isa_circuits, shots = nshots)
        job_id = simjob.job_id()
        
    return job_id
        
    

        

def getResultsBatch(jobs_pub_results):
    countsArr = []
    for pub in jobs_pub_results:
        counts = pub.data.meas.get_counts()
        counts.setdefault('0', 0)
        counts.setdefault('1', 0)
        countsArr.append(counts['1']/nshots)

    return countsArr




def simulate_single_circuit(qc,n_shots):
    return simulateAerBackend(qc, n_shots)

