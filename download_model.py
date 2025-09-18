# save as: download_nf4_here.py (run it from your project folder)
from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    local_dir = Path.cwd() / "MolmoE-1B-0924-NF4"  # will be created in the folder you're in
    local_dir.mkdir(parents=True, exist_ok=True)

    p = snapshot_download(
        repo_id="reubk/MolmoE-1B-0924-NF4",
        local_dir=str(local_dir),           # download right here
        local_dir_use_symlinks=False,       # write real files
        revision="main",
    )
    print("Saved to:", p)

if __name__=='__main__':
    main()