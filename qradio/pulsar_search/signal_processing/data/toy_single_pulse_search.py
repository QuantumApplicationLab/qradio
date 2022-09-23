#!/usr/bin/env python3
import numpy as np
from scipy.signal import gaussian
import matplotlib.pyplot as plt
import tqdm


def generate_data(nchan, ntime, period, dm, width, noise_level=0.):
    """Generate fake pulsar data with somewhat realistic pulse profile

    Args:
        nchan (int): Number of frequency channels
        ntime (int): Number of time samples
        period (float): Pulsar period in units of time samples
        width (float): Pulse width in units of time samples (std of Gaussian)
        dm (float): DM in units of time samples across band
        noise_level (float): Amount of noise to add (std of gaussian dist)
    Returns:
        Raw data with shape (nchan, ntime)
    """
    # generate reference timeseries with gaussian pulses
    profile = gaussian(period, width)
    # add extra periods to be cut off later, so pulses that are
    # only partially within the band are preserved
    num_extra_bins = int(dm / period + 1) * period
    ntime_raw = ntime + num_extra_bins
    num_pulses = int((ntime_raw / period) + 1)
    print(f"Number of pulses completely in band: {int(ntime / period)}")
    timeseries = np.array(list(profile) * num_pulses)[:ntime_raw]

    # apply dispersion to generate data as it would be detected by a telescope
    data = np.zeros((nchan, ntime_raw))
    for channel_index in range(nchan):
        shift = int(dm * ((nchan - 1 - channel_index) / (nchan - 1)) ** 2)
        data[channel_index, shift:] = timeseries[:ntime_raw-shift]

    data = data[:, num_extra_bins:]
    # Add noise
    data += np.random.normal(scale=noise_level, size=data.shape)

    return data


def dedisperse(data, dm):
    """Dedisperses data to the given dispersion measure.

    Args:
        data (array): (nchan, ntime) array with raw data
        dm (float): dispersion measure in units of time samples across band

    Returns:
        Timeseries of size ntime
    """
    nchan, ntime = data.shape
    timeseries = np.zeros(ntime)
    for channel_index, channel in enumerate(data):
        shift = int(dm * ((nchan - 1 - channel_index) / (nchan - 1)) ** 2)
        timeseries += np.roll(channel, -shift)
    return timeseries


def get_snr_with_filter(data, width):
    """Apply a boxcar filter to the input data then
       calculate the S/N at each time point

    Args:
        data (array): input data (1D array)
        width (int): filter width

    Returns:
        S/N values
    """
    # filter step
    matched_filter = np.ones(width)
    data_filtered = np.correlate(data, matched_filter)
    # S/N step
    return (data_filtered - np.median(data_filtered)) / data.std()


def make_plot(data, x, y):
    """
    Make a plot of single pulse candidates

    Args:
        data (array): numpy array with named columns
        x (str): column to use for x axis
        y (str): column to use for y axis
    """

    plt.figure()
    plt.scatter(data[x], data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('candidates')


if __name__ == '__main__':
    # data characteristics
    nchan = 128
    ntime = 2048
    noise_level = .5

    # pulsar parameters
    psr_period = 500
    psr_width = 4
    psr_dm = 250

    # search parameters
    dms = range(0, 1000, 2)
    widths = range(1, 10)
    snr_threshold = 50

    raw_data = generate_data(nchan, ntime, psr_period, psr_dm, psr_width, noise_level)

    pulse_candidates = []
    for dm in tqdm.tqdm(dms, desc="Single pulse search"):
        timeseries = dedisperse(raw_data, dm)
        for width in widths:
            # boxcar filter
            snr_all = get_snr_with_filter(timeseries, width)
            # store where above S/N threshold
            time_steps = np.where(snr_all > snr_threshold)[0]
            for time in time_steps:
                # store the candidate
                # candidate = [dm, time, width, snr]
                snr = snr_all[time]
                candidate = f"{dm:.2f},{time+.5*width:.0f},{width:.0f},{snr:.2f}\n"
                pulse_candidates.append(candidate)

    print(f"Found {len(pulse_candidates)} pulse candidates")

    with open('single_pulse_candidates.csv', 'w') as f:
        f.write("dm,time,width,snr\n")
        f.writelines(pulse_candidates)

    # plot data
    plt.figure()
    plt.imshow(raw_data, origin='lower', aspect='auto')
    plt.xlabel('time')
    plt.ylabel('channels')
    plt.title('Raw data')

    # plot timeseries at pulsar DM
    plt.figure()
    plt.plot(dedisperse(raw_data, psr_dm))
    plt.xlabel('time')
    plt.ylabel('power')
    plt.title('Timeseries at pulsar DM')

    # plot some 2D slices of the candidates
    sps_data = np.genfromtxt('single_pulse_candidates.csv', delimiter=',', names=True)
    make_plot(sps_data, 'time', 'dm')
    make_plot(sps_data, 'dm', 'snr')
    make_plot(sps_data, 'width', 'snr')
    plt.show()
