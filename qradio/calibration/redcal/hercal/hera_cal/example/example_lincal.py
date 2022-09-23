
import numpy as np
from copy import deepcopy
import warnings
import os
import sys
import shutil
from hera_sim.antpos import linear_array, hex_array
from hera_sim.vis import sim_red_data
from hera_sim.sigchain import gen_gains

from hera_cal import redcal as om
from hera_cal import io, abscal
from hera_cal.utils import split_pol, conj_pol, split_bl
from hera_cal.apply_cal import calibrate_in_place
from hera_cal.data import DATA_PATH
from hera_cal.datacontainer import DataContainer


from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer

from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit_research.vqls.runtime.runtime_wrapper import vqls_runner

# ibmq_token = ''
# hub = 'ibm-q-qal'
# group = 'escience' # examnple 'escience'
# project = 'qradio' # example qradio

# QiskitRuntimeService.save_account(channel="ibm_quantum", 
#                                   token=ibmq_token, 
#                                   instance=hub+'/'+group+'/'+project,
#                                   overwrite=True)

# service = QiskitRuntimeService()

# program_id = 'vqls-7WKEvMkw2V'

# backend = service.backend("simulator_statevector")
# job = vqls_runner(backend, A, b, program_id, ansatz, shots=25000)
# res = job.result()

NANTS = 4
NFREQ = 64
antpos = linear_array(NANTS)
reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
info = om.RedundantCalibrator(reds)
fqs = np.linspace(.1, .2, NFREQ)
g, true_vis, d = sim_red_data(reds, shape=(1, NFREQ), gain_scatter=0)
delays = {k: np.random.randn() * 30 for k in g.keys()}  # in ns
fc_gains = {k: np.exp(2j * np.pi * v * fqs) for k, v in delays.items()}
delays = {k: np.array([[v]]) for k, v in delays.items()}
fc_gains = {i: v.reshape(1, NFREQ) for i, v in fc_gains.items()}
gains = {k: v * fc_gains[k] for k, v in g.items()}
gains = {k: v.astype(np.complex64) for k, v in gains.items()}
calibrate_in_place(d, gains, old_gains=g, gain_convention='multiply')
d = {k: v.astype(np.complex64) for k, v in d.items()}


info.set_quantum_backend(Aer.get_backend('aer_simulator_statevector'))
num_qubits = int(np.ceil(np.log2(NANTS)))
info.set_quantum_ansatz(RealAmplitudes(num_qubits, entanglement='full' , reps=3, insert_barriers=False))
info.set_quantum_optimizer(COBYLA(maxiter=50, disp=True))



dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], 
                                            medfilt=False, 
                                            mode='vqls')
sol_degen = info.remove_degen_gains(dly_sol, degen_gains=delays, mode='phase')
for i in range(NANTS):
    assert dly_sol[(i, 'Jxx')].dtype == np.float64
    assert dly_sol[(i, 'Jxx')].shape == (1, 1)
    assert np.allclose(np.round(sol_degen[(i, 'Jxx')] - delays[(i, 'Jxx')], 0), 0)