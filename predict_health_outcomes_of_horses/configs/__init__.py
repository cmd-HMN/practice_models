from .CFG import CFG
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

cfg = CFG()
console = Console()

if cfg.debug:
    cfg.update_debug()
    console.print("Debug mode activated! Adjusting configuration.\n", style="bold yellow")