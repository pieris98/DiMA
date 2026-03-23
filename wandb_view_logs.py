import glob
import os
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2

# 1. Resolve the path to the .wandb file
# 'latest-run' is usually a symlink created by wandb in your project folder
base_path = "./wandb/latest-run/*.wandb"
files = glob.glob(base_path)

if not files:
    print(f"No .wandb files found in {base_path}. Check your path!")
else:
    # Usually, there is only one .wandb file per run folder
    data_path = files[0]
    print(f"Reading: {data_path}")

    ds = datastore.DataStore()
    ds.open_for_scan(data_path)

    while True:
        data = ds.scan_record()
        if data is None: 
            break  # End of file
        
        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data[1])
        
        # Check the record type (history contains the metrics)
        if pb.WhichOneof("record_type") == "history":
            for item in pb.history.item:
                # Use item.key for the metric name and item.value_json for the data
                print(f"Key: {item.key}, Value: {item.value_json}")