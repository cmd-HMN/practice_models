import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import zipfile
import subprocess
from configs import cfg
from argparse import ArgumentParser
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.align import Align

console = Console()


def check_dataset(path=(cfg.BASE_PATH)):
    return any(fname.endswith(".csv") for fname in os.listdir(path)) if os.path.exists(path) else False


def download_and_unzip():
    os.makedirs(cfg.BASE_PATH, exist_ok=True)

    console.print("[yellow]Downloading dataset from Kaggle...[/yellow]\n")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "playground-series-s3e22", "-p", cfg.BASE_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except:
        console.log("Error", style="bold red")
        table = Table(box=box.SQUARE, style="bold red")
        table.add_row("Visit [link=https://www.kaggle.com/settings]kaggle.com/settings[/link] to download your kaggle.json or download dataset [link=https://www.kaggle.com/competitions/playground-series-s3e22/data]manually[/link]")
        console.print(table)
        return

    zip_files = [f for f in os.listdir(cfg.BASE_PATH) if f.endswith(".zip")]

    for file in track(zip_files, description="Unzipping files..."):
        zip_path = os.path.join(cfg.BASE_PATH, file)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cfg.BASE_PATH)
            os.remove(zip_path)
        except Exception as e:
            console.print(f"[red]Failed to unzip {file}: {e}[/red]")

    console.print("[bold green]Dataset ready in data folder.[/bold green]")

def main():
    parser = ArgumentParser(description="Download Manager")
    parser.add_argument("-c", "--check", action="store_true", help="Check if dataset is already downloaded")
    parser.add_argument("-i", "--install", action="store_true", help="Download and unzip dataset from Kaggle")
    parser.add_argument("-ci", "--check-install", action="store_true", help="Check and install the dataset")

    args = parser.parse_args()

    if args.check:
        console.log("Checking")
        exists = check_dataset()
        console.print(f"[cyan]Dataset exists:[/cyan] [bold]{exists}[/bold]")
        return exists

    if args.install:
        if check_dataset():
            console.print("[green]Dataset already exists. Skipping download.[/green]")
        else:
            download_and_unzip()

    if args.check_install:
        console.log("Checking")
        if check_dataset():
            console.print("[green]Dataset already exists. Skipping download.[/green]")
        else:
            console.print("[red]Not Present.[/red]")
            download_and_unzip()

if __name__ == "__main__":
    main()