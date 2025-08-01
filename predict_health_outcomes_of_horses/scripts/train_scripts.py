import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import cfg
from rich.console import Console
import time

from train import train_find_weights
from code_base import preprocessing, read_data

console = Console()

with console.status("[bold]Loading Dataset in data directory...[/bold]", spinner="dots"):
    time.sleep(1)
    try:
        train = read_data("train")
    except Exception as e:
        console.print(f"May be the files aren't in the {cfg.BASE_PATH}")
        exit()


console.print("\n[bold green]Loaded the dataset![/bold green]\n")


with console.status(f"[bold]Preprocessing train dataset[/bold]", spinner="arc"):
    time.sleep(1)

    train.drop(['id', 'lesion_3'], axis=1, inplace=True)
    train.drop_duplicates(inplace=True)

    X = train.drop('outcome', axis=1)
    y = train['outcome'].map({'died':0, 'lived':1, 'euthanized':2})

    X = preprocessing(X, le_cols=cfg.binary_col, ohe_cols=cfg.ohe_cols)


console.print("\n[bold green]Preprocessing completed![/bold green]\n")

console.log("[bold]Training Model\n[/bold]")
train_find_weights(X, y)


console.print("\n[bold green]Model Training Completed![/bold green]\n")