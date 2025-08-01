import subprocess

if __name__ == "__main__":
    try:
        process = [
            ["python", "scripts/data_scripts.py", "-ci"],
            ["python", "scripts/train_scripts.py"],
            ["python", "scripts/predict_scripts.py"],
        ]

        for p in process:
            subprocess.run(p, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed: {e}")