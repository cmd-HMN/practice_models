from code_base import Splitter, Classifier, OptunaWeights, save_file, score
from configs import cfg
import numpy as np
import sys
import tqdm
from copy import deepcopy


from rich.console import Console
from rich.progress import track
from rich.progress import Progress
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.align import Align

def train_find_weights(X, y, cfg=cfg):
    
    console = Console()

    ensemble_score = []
    ensemble_f1_score = []
    weights = []
    trained_models = {'svc': [],
    'xgb': [], 'xgb2': [],
    'lgb': [], 'lgb2': [], 'lgb3': [],
    'cat': [], 'cat_sym': [], 'cat_loss': [], 'cat2': [],
    'rf': [],
    'hist_gbm': []}
    
    splitter = Splitter(cfg=cfg)
    for i, (X_train, X_val, y_train, y_val) in enumerate(splitter.split_data(X, y)):
        
        classifier = Classifier('main', cfg)
        models = classifier.models

        oof_preds = []
        test_preds = []

        max_score = -1
        mname = ''

        console.print(f"\n====================== Fold {i + 1} ======================\n", style="bold dark_violet")        
        text = ''

        with Progress(
                "[progress.description]{task.description}",
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",    
                "[purple]{task.completed}/{task.total} models",
                transient=True,
            ) as progress:

            task = progress.add_task("[bright_magenta]Training Models...", total=len(models))

            for name, model in models.items():
                if 'xgb' in name or 'cat' in name:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=cfg.verbose)
                elif 'lgb' in name:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    model.fit(X_train, y_train)

                trained_models[name].append(deepcopy(model))

                y_val_pred = model.predict_proba(X_val)
                y_val_pred_labels = np.argmax(y_val_pred, axis=1)
                f1_micro_score = score(y_val, y_val_pred_labels)
                oof_preds.append(y_val_pred)

                text += f"[cyan]{name}[/cyan] ----- F1 Micro Score: [bold yellow]{f1_micro_score:.5f}[/bold yellow]\n"

                if f1_micro_score > max_score:
                    max_score = f1_micro_score
                    mname = name

                progress.advance(task)


        console.print(
                Panel.fit(
                    f"Best Model: [bold cyan]{mname}[/bold cyan]\nBest Score: [bold yellow]{max_score:.5f}[/bold yellow]",
                    title=f"Fold {i + 1} Result",
                    border_style="red3"
                ), new_line_start=True
            )

        
        with console.status(f"[bold]Optuna on the run[/bold]", spinner="arc"):
            optweights = OptunaWeights(cfg)
            y_val_pred = optweights.fit_predict(y_val, oof_preds)
        
            y_val_pred_labels = np.argmax(y_val_pred, axis=1)
            f1_micro_score = score(y_val, y_val_pred_labels)
            text += f'[bold red]Ensemble Score[/bold red]  ----- F1 Micro Score: [bold yellow]{f1_micro_score:.5f}[/bold yellow]'

        console.print(Panel.fit(text, title="F1 Scores", border_style="purple3"), new_line_start=True)

        ensemble_score.append(score)
        ensemble_f1_score.append(f1_micro_score)
        weights.append(optweights.weights)  

    save_file('trained_models.pkl', trained_models)
    save_file('ensemble_score.pkl', ensemble_score)
    save_file('weights.pkl', weights)
    save_file('ensemble_f1_score.pkl', ensemble_f1_score)
    
    return trained_models, weights, ensemble_score, ensemble_f1_score
