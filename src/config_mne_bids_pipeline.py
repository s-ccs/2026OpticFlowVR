from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

bids_root = PROJECT_ROOT / "data_clean"

deriv_root = (
    PROJECT_ROOT
    / "output"
    / "derivatives"
    / "mne-bids-pipeline"
)

subjects = [
    "002", 
    "003", 
    "004", 
    "005", 
    "006", 
    "007", 
    "008", 
    "009", 
    "010", 
    "011", 
    "012",
    "013",
    "014",
    "015",
]
sessions = ["001"]
task = "compareSpeed"
runs = ["1"]

eeg_template_montage = "standard_1020"
ch_types = ["eeg"]
eeg_reference = "average"

l_freq = 0.1
h_freq = 100.0
notch_freq = 50.0
raw_resample_sfreq = 250
plot_psd_for_runs = [] # disable PSD plotting

spatial_filter = None
# ica_reject = "autoreject_local"

conditions = [
    "Forward",
    "Random",
    "Rotation/Left",
    "Rotation/Right",
    "Spiral/Left",
    "Spiral/Right",
]

epochs_tmin = -0.2
epochs_tmax = 0.8
reject = {"eeg": 200e-6}
