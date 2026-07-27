from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SESSION = "001"
TASK = "compareSpeed"
RUN = "1"

DERIV_ROOT = (
    PROJECT_ROOT / "output" / "derivatives" / "mne-bids-pipeline-iclabel"
)
CLEAN_BIDS_ROOT = PROJECT_ROOT / "data_clean"
OUT_ROOT = PROJECT_ROOT / "output-iclabel" / "tfa"

# Keep synchronized with the final thesis inclusion list.
SUBJECTS = [
    "002", "003", "004", "005", "006", "007", "008", "009",
    "010", "011", "012", "013", "014", "015", "016",
    # "017",  # unavailable
    "018", "019", "020", "021", "022", "023", "024",
    "026", "027", "029", "030", "031", "032", "034", "035",
]

POSTERIOR_ROI = ["O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"]

# Long epochs include padding for wavelet convolution.
# Reported/visualized analyses are restricted to ANALYSIS_TMIN..ANALYSIS_TMAX.
EPOCH_TMIN = -1.20
EPOCH_TMAX = 2.50
ANALYSIS_TMIN = -0.60
ANALYSIS_TMAX = 2.10

BASELINE = (-0.50, -0.10)
BASELINE_MODE = "logratio"  # natural log(power / baseline mean)

# Primary exploratory range.
FREQS = np.arange(3.0, 81.0, 1.0)

# At least 3 cycles at low frequencies; cap at 10 cycles at high frequencies.
# Approximate wavelet duration is n_cycles / frequency.
N_CYCLES = np.clip(FREQS / 2.0, 3.0, 10.0)

# Raw is resampled to 250 Hz in the MNE-BIDS-Pipeline.
# decim=2 produces a TFR time step of 8 ms after convolution.
TFR_DECIM = 2

# Same EEG peak-to-peak criterion as the preprocessing configuration.
REJECT = {"eeg": 300e-6}

CONDITION_ORDER = ["Forward", "Random", "Rotation", "Spiral"]
CONTRASTS = {
    "Random-minus-Forward": ("Random", "Forward"),
    "Rotation-minus-Forward": ("Rotation", "Forward"),
    "Spiral-minus-Forward": ("Spiral", "Forward"),
}

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
}
TIME_WINDOWS = {
    "early": (0.0, 0.5),
    "sustained": (0.5, 2.0),
}
