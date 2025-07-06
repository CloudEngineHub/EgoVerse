import os
import subprocess
import boto3
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter.filedialog import askdirectory
import pandas as pd
from pathlib import Path

# AWS S3 setup
s3 = boto3.client("s3")

# Root window for simpledialog prompts
tk_root = tk.Tk()
tk_root.withdraw()

# Ask user for required information
task_name = simpledialog.askstring("Task Name", "Enter task name (e.g., pour_milk):")
if not task_name:
    raise ValueError("Task name is required.")
s3_base_path = f"s3://rldb/raw/{task_name}/"

folder_path = askdirectory(title="Select folder containing .vrs and .vrs.json files")
if not folder_path:
    raise ValueError("Folder selection is required.")
folder_path = Path(folder_path)

# Ask for custom metadata fields
custom_keys_str = simpledialog.askstring("Custom Metadata", "Enter additional metadata keys (comma-separated):")
custom_keys = [k.strip() for k in custom_keys_str.split(",")] if custom_keys_str else []

# Default metadata fields
default_keys = ["collector", "lab", "scene", "recording_number"]
all_keys = default_keys + custom_keys

# List to track files for deferred upload
upload_queue = []

# Process each .vrs file
vrs_files = sorted(folder_path.glob("*.vrs"))
for vrs_file in vrs_files:
    json_file = vrs_file.with_suffix(".vrs.json")
    if not json_file.exists():
        print(f"Missing .vrs.json for {vrs_file}, skipping.")
        continue

    # Launch vrsplayer
    player_proc = subprocess.Popen(["vrsplayer", str(vrs_file)])

    # GUI for metadata input
    metadata_window = tk.Tk()
    metadata_window.title(f"Metadata for {vrs_file.name}")
    entries = {}
    submitted_metadata = {}

    def submit_and_close():
        for key in all_keys:
            submitted_metadata[key] = entries[key].get()
        metadata_window.quit()  # End mainloop cleanly

    for idx, key in enumerate(all_keys):
        tk.Label(metadata_window, text=key).grid(row=idx, column=0)
        e = tk.Entry(metadata_window, width=40)
        e.grid(row=idx, column=1)
        entries[key] = e

    tk.Button(metadata_window, text="Submit", command=submit_and_close).grid(row=len(all_keys), columnspan=2)

    metadata_window.mainloop()
    metadata_window.destroy()

    # Now continue processing
    player_proc.terminate()

    metadata = submitted_metadata
    lab = metadata["lab"]
    scene = metadata["scene"]
    rec = metadata["recording_number"]
    new_name_base = f"{task_name}_{lab}_{scene}_recording_{rec}"

    renamed_vrs = folder_path / f"{new_name_base}.vrs"
    renamed_json = folder_path / f"{new_name_base}.vrs.json"
    metadata_csv = folder_path / f"{new_name_base}_meta.csv"
    
    os.rename(vrs_file, renamed_vrs)
    print(f"Renamed {vrs_file} to {renamed_vrs}")
    os.rename(json_file, renamed_json)
    print(f"Renamed {json_file} to {renamed_json}")
    pd.DataFrame([metadata]).to_csv(metadata_csv, index=False)
    print(f"Saved metadata to {metadata_csv}")

    # Defer upload
    upload_queue.extend([renamed_vrs, renamed_json, metadata_csv])

# Now upload everything
print("\nUploading all processed files to S3...")
for file_path in upload_queue:
    s3_path = f"raw/{task_name}/{file_path.name}"
    print(f"Uploading {file_path.name} to s3://rldb/{s3_path}")
    s3.upload_file(str(file_path), "rldb", s3_path)

messagebox.showinfo("Done", "All VRS files processed and uploaded.")
