import numpy as np
from qiskit import QuantumCircuit

def integrate_signal(data, n_count, shift, incr):
    bits = []
    ntime = data.shape[0]
    for icol in range(ntime):
        row_idx_start = shift
        row_idx_end = shift+n_count*incr
        d = data[row_idx_start:row_idx_end:incr,icol]
        bits.append(BitsToIntAFast(d, invert_bits_order=True))
    return bits

def BitsToIntAFast(bits, invert_bits_order=True):
    n = len(bits)  # number of columns is needed, not bits.size
    # -1 reverses array of powers of 2 of same length as bits
    a = 2**np.arange(n)
    if invert_bits_order:
        a = a[::-1]
    return bits @ a  # this matmult is the key line of code


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def tobin(x, length):
    xbin = bin(x).replace('0b','')
    while len(xbin)<length:
        xbin = '0'+xbin
    return xbin

def generate_fake_data(nchan, ntime, period, dm, noise_frac=.1):
    """Create an (nchan x ntime) array with simulated pulsar signal
       nchan (int): Number of frequency channels
       ntime (int): Number of time samples
       period (float): Pulsar period in units of time samples
       dm (float): DM in units of time samples across band
    """
    assert (ntime / period) % 1 == 0, "for now need integer periods in ntime"
    data = np.zeros((nchan, ntime), dtype=np.uint8)
    # get indices where pulsar is "on" in reference frequency channel (= highest frequency = last channel)
    nperiod = int(ntime / period) + 1
    pulsar_indices_highest_freq = (np.arange(nperiod) * period).astype(int)
    # calculate DM shifts for each channel (no shifts at highest channel, positive towards lower channels)
    for chan in range(nchan):
        # DM shift of this channel
        shift = dm * ((nchan - 1 - chan) / (nchan - 1)) ** 2
        # pulsar location in this channel
        pulsar_indices = (pulsar_indices_highest_freq + shift) % ntime  # without % ntime if not integer nr of periods in ntime
        data[chan, pulsar_indices.astype(int)] = 1

    # add some random noise, i.e. set given fraction of data to 1
    data += (np.random.random(data.shape) < noise_frac)
    return data


def generate_fake_data_2(nchan, ntime, period, dm, noise_frac=.1):
    """Create an (nchan x ntime) array with simulated pulsar signal
       nchan (int): Number of frequency channels
       ntime (int): Number of time samples
       period (float): Pulsar period in units of time samples
       dm (float): DM in units of time samples across band
    """
    # assert (ntime / period) % 1 == 0, "for now need integer periods in ntime"
    data = np.zeros((nchan, ntime), dtype=np.uint8)
    # get indices where pulsar is "on" in reference frequency channel (= highest frequency = last channel)
    nperiod = int(ntime / period) + 1
    pulsar_indices_highest_freq = (np.arange(nperiod) * period).astype(int)
    # calculate DM shifts for each channel (no shifts at highest channel, positive towards lower channels)
    for chan in range(nchan):
        # DM shift of this channel
        shift = dm * ((nchan - 1 - chan) / (nchan - 1)) ** 2
        # pulsar location in this channel
        # print(shift)
        pulsar_indices = (pulsar_indices_highest_freq + shift) #% ntime  # without % ntime if not integer nr of periods in ntime
        # print(pulsar_indices)
        pulsar_indices = pulsar_indices.astype(int)
        pulsar_indices = list(pulsar_indices)
        while pulsar_indices[-1]>=ntime:
            pulsar_indices.pop()

        while pulsar_indices[0]>=period:
            pulsar_indices = [pulsar_indices[0]-period] + pulsar_indices

        pulsar_indices = np.array(pulsar_indices)
        data[chan, pulsar_indices] = 1

    # add some random noise, i.e. set given fraction of data to 1
    data += (np.random.random(data.shape) < noise_frac)
    return data