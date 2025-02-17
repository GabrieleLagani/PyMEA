from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

import utils


PATH = 'results/mnist/iter0/recordings0/0/CHIR_N1_DIV30_25928.pt'

f = 20000.0 # Frequency of the recording.
start, w = int(1000e-3 * f), int(2000e-3 * f) # Start time and window size to plot.
stimtime = int(2000e-3 * f) - start # Time of the stimulus in the recording.
v_thr = 400  # None to disable. Default 400. Threshold over voltage to exclude spikes associated with excessive peak-to-peak voltage (possible electrical artifacts).
w_thr = 15 # None to disable. Default 15. Threshold over temporal duration to exclude spikes associated with excessive temporal extension (possible electrical artifacts).

nh, nw = 64, 64
rows, cols = 8, 8
figsize = (20, 12)


def peak2peak(sf):
    return (sf.max() - sf.min()).item()

def amplitude(sf):
    return sf.abs().max().item()

def spikewidth(sf):
    return sf.abs().sum().item()/sf.abs().max().item()

def filter_spikes(spike_times, channels, spike_forms):
    idx = [True for _ in range(len(spike_times))]
    #if v_thr is not None: idx = [idx[i] and peak2peak(sf) < v_thr for i, sf in enumerate(spike_forms)]
    if v_thr is not None: idx = [idx[i] and amplitude(sf) < v_thr for i, sf in enumerate(spike_forms)]
    if w_thr is not None: idx = [idx[i] and spikewidth(sf) < w_thr for i, sf in enumerate(spike_forms)]
    return [spike_times[i] for i, selected in enumerate(idx) if selected], [channels[i] for i, selected in enumerate(idx) if selected], [spike_forms[i] for i, selected in enumerate(idx) if selected]

def select_spike_forms(spike_times, channels, spike_forms):
    def _select_spike_forms_top(spike_times, channels, spike_forms):
        _, spike_times, channels, spike_forms = list(
            zip(*sorted([(peak2peak(sf), st, ch, sf) for st, ch, sf in zip(spike_times, channels, spike_forms)],
                        key=lambda x: x[0])))
        k = min(rows * cols, len(spike_forms))
        spike_times, channels, spike_forms = spike_times[-k:], channels[-k:], spike_forms[-k:]
        return spike_times, channels, spike_forms

    def _select_spike_forms_first(spike_times, channels, spike_forms):
        k = min(rows * cols, len(spike_forms))
        spike_times, channels, spike_forms = spike_times[:k], channels[:k], spike_forms[:k]
        return spike_times, channels, spike_forms

    return _select_spike_forms_top(spike_times, channels, spike_forms)


if __name__ == '__main__':
    data_dict = utils.load_dict(PATH, device='cpu')
    data = data_dict['raw']
    spike_times, channels, spike_forms = data_dict['processed']['spike_times'], data_dict['processed']['channels'], data_dict['processed']['spike_forms']
    idx = [i for i, st in enumerate(spike_times) if st > start and st < start + w]
    spike_times, channels, spike_forms = [spike_times[i] for i in idx], [channels[i] for i in idx], [spike_forms[i] for i in idx]
    spike_times, channels, spike_forms = filter_spikes(spike_times, channels, spike_forms)

    if data is not None:
        print("Plotting pooled activity")
        data = data[start:start + w]
        bh, bw = nh // rows, nw // cols
        data = data.reshape(-1, rows, bh, cols, bw).mean(axis=4).mean(axis=2)
        #data = data[:, :2, :2]
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False, figsize=figsize)
        for r in range(rows):
            for c in range(cols):
                if c >= data.shape[2] or r >= data.shape[1]: continue
                axs[r][c].plot(list(range(start, start+w)), data[:, r, c].numpy())
                x = [spike_times[i] for i in range(len(channels)) if (channels[i]%nw) // cols == r and (channels[i]//nw) // rows == c]
                y = [spike_forms[i].min(dim=0)[0] for i in range(len(channels)) if (channels[i]%nh) // cols == r and (channels[i]//nh) // rows == c]
                axs[r][c].scatter(x, y, color='red')
                if c == 0: axs[r][c].set_ylabel("uV")
                if r == rows - 1: axs[r][c].set_xlabel("Frame")
        fig.tight_layout()

    print("Plotting spike correlation")
    spikes_per_channels = np.zeros(nh*nw, dtype=np.float32)
    for ch in tqdm(range(nh*nw), ncols=80):
        spikes_ch = [st for i, st in enumerate(spike_times) if channels[i] == ch]
        spikes_per_channels[ch] = len([st for st in spikes_ch if st >= stimtime]) - len([st for st in spikes_ch if st < stimtime])
    nrm_top, nrm_btm = np.max(spikes_per_channels), np.min(spikes_per_channels)
    grid = (spikes_per_channels - nrm_btm) / (nrm_top - nrm_btm)
    fig = plt.figure(figsize=figsize)
    fig.gca().imshow(grid.reshape(nh, nw))

    print("Plotting selected spike forms")
    spike_times, channels, spike_forms = select_spike_forms(spike_times, channels, spike_forms)
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            if r*cols + c >= len(spike_forms): continue
            sf = spike_forms[r*cols + c]
            axs[r][c].plot(sf.numpy())
            axs[r][c].set_title("t={}, ch={}, \n p2p={:.1f}, ampl={:.1f}, width={:.1f}".format(
                spike_times[r*cols + c], channels[r*cols + c], peak2peak(sf), amplitude(sf), spikewidth(sf)), fontsize=6)
            if c == 0: axs[r][c].set_ylabel("uV")
            if r == rows - 1: axs[r][c].set_xlabel("Frame")
    fig.tight_layout()
    plt.show()


