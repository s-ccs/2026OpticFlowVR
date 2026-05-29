import pyxdf

streams, _ = pyxdf.load_xdf(
    "data/sub-001/ses-001/eeg/sub-001_ses-001_task-Default_run-001_eeg_raw.xdf"
)

for i, stream in enumerate(streams):
    print("\n--- STREAM", i, "---")
    print("name:", stream["info"]["name"][0])
    print("type:", stream["info"]["type"][0])
    print("nominal_srate:", stream["info"]["nominal_srate"][0])
    print("channels:")

    desc = stream["info"].get("desc", [])
    print(desc)