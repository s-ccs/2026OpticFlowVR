from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

bids_root = PROJECT_ROOT / "data_clean"

deriv_root = (
    PROJECT_ROOT
    / "output"
    / "derivatives"
    / "mne-bids-pipeline-iclabel"
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
    "016",
    # "017",  # could not be converted to BIDS
    "018",
    "019",
    "020",
    "021",
    "022",
    "023",
    "024",
    "026",
    "027",
    "029",
    "030",
    "031",
    "032",
    "034",
    "035",
    "036",
    "037",
    "038",
    "039",
    "040",
]
sessions = ["001"]
task = "compareSpeed"
runs = ["1"]

base_conditions = [
    "Forward",
    "Random",
    "Rotation/Left",
    "Rotation/Right",
    "Spiral/Left",
    "Spiral/Right",
]

conditions = [
    f"{condition}/speed-{speed_idx}"
    for condition in base_conditions
    for speed_idx in range(7)
]

eeg_template_montage = "standard_1020"
ch_types = ["eeg"]
eeg_reference = "average"

l_freq = 0.1
h_freq = 100.0
notch_freq = 50.0
raw_resample_sfreq = 250

spatial_filter = "ica"

ica_use_icalabel = True
ica_l_freq = 1.0
ica_h_freq = 100.0
ica_algorithm = "picard-extended_infomax"
ica_reject = {"eeg": 600e-6}
reject = {"eeg": 300e-6}

epochs_tmin = -0.2
epochs_tmax = 1.2
