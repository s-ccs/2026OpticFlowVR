# Import the necessary libraries
import mne
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os 
import numpy as np

sample_data_folder = './data/'
sample_data_raw_file = os.path.join(sample_data_folder, 'sub-002', 'ses-001', 'eeg',
                                    'sub-002_ses-001_task-compareSpeed_run-1_eeg.set')
raw = mne.io.read_raw_eeglab(sample_data_raw_file, preload=True)

raw.plot(block=True);
