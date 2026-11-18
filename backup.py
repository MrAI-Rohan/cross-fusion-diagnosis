"""Copy a finished run's best checkpoint to cloud storage with rclone.

Set the destination once in the shell before training, for example:
    export CKPT_BACKUP_REMOTE=gdrive:recall-cure/checkpoints
The files end up in <remote>/<dataset>/<experiment_name>/ (the best checkpoint and config.yaml).
If CKPT_BACKUP_REMOTE is not set, nothing is copied and the run only prints a notice.
The source folder is only read, never changed. A failed backup is printed loudly but does not raise, so a
finished training run is never reported as failed because of a network problem.
"""
import os
import shutil
import subprocess
from pathlib import Path


def backup_checkpoint(checkpoint_dir, best_model_path, experiment_name, dataset_name, timeout_s=3600):
    remote = os.environ.get("CKPT_BACKUP_REMOTE")
    if not remote:
        print("[backup] CKPT_BACKUP_REMOTE is not set: the best checkpoint was NOT backed up.")
        return False
    if not best_model_path or not Path(best_model_path).is_file():
        print(f"[backup] BACKUP FAILED: no best checkpoint file found ({best_model_path!r}).")
        return False
    if shutil.which("rclone") is None:
        print("[backup] BACKUP FAILED: rclone is not installed or not on PATH.")
        return False

    dest = f"{remote.rstrip('/')}/{dataset_name}/{experiment_name}"
    cmd = ["rclone", "copy", str(checkpoint_dir), dest,
           "--include", Path(best_model_path).name, "--include", "config.yaml",
           "--retries", "3", "--stats-one-line", "-v"]
    print(f"[backup] copying {Path(best_model_path).name} to {dest} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as e:                       # timeout, OS error, anything else
        print(f"[backup] BACKUP FAILED: {type(e).__name__}: {e}")
        return False
    if result.returncode != 0:
        print(f"[backup] BACKUP FAILED (rclone exit code {result.returncode}):\n{result.stderr.strip()}")
        return False
    print(f"[backup] done: {dest}")
    return True