from qiskit_ibm_runtime import QiskitRuntimeService

# Save yout account 
QiskitRuntimeService.save_account(
        channel='ibm_cloud', 
        token='byjZdOKqi21vFjVL69lZkqAeyQJ1bdWsror1wPcmuT3S',
        instance='crn:v1:bluemix:public:quantum-computing:eu-de:a/287d3331d81a44dd8507160b72f4c0fc:c8d942c1-df8d-47ea-a4d3-854f423ff2b2::',
        name='qmlserver', 
        set_as_default=True)
