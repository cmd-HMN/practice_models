import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import cfg
from rich.console import Console
import time

from inference import predict, create_submission

console  = Console()

console.print("[bold]\nMaking Prediction\n[/bold]")

preds = predict()

create_submission()


console.print("\n[bold green]Every thing wents well the prediction are in the data dir![/bold green]\n")
