from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(script_path: str) -> None:
    full_path = PROJECT_ROOT / script_path
    print(f"\nRunning {script_path}", flush=True)
    subprocess.run([sys.executable, str(full_path)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    run_step("code/datasets/download_data.py")
    run_step("code/datasets/prepare_data.py")
    run_step("code/models/train_model.py")
    print("\nPipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
