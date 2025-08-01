from code_base import load_file, preprocessing, read_data
import tqdm
import numpy as np
import sys
from configs import cfg
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import track
from rich.progress import Progress
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.align import Align



def predict(test=None, trained_models=None, weights=None, cfg=cfg):
    console = Console()
    
    if test == None:
        test = read_data("test")
        cfg.id_ = test.id
        test.drop(['id', 'lesion_3'], axis=1, inplace=True)
        test.drop_duplicates(inplace=True)

    if trained_models == None:
        trained_models = load_file('trained_models.pkl')

    if weights == None:
        weights = np.array(load_file('weights.pkl'))

    test = preprocessing(test, le_cols=cfg.binary_col, ohe_cols=cfg.ohe_cols)

    models = list(trained_models.keys())
    n_folds = len(weights)
    n_models = len(models)

    # store the fold result
    test_preds = np.zeros((test.shape[0], cfg.n_classes))

    assert weights.shape == (n_folds, n_models), "Weights shape mismatch!"
    
    with Progress(
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            "[purple]{task.completed}/{task.total} models",
            transient=True,
        ) as progress:

        task = progress.add_task("[bright_magenta]Folds...", total=len(weights))
        for i, w in enumerate(weights):           
            pbar = enumerate(models)

            pt = progress.add_task("[bright_magenta]Evaluating Model...", total=len(models))
            # store the model result here
            fold_weights = weights[i]
            ttest = np.zeros_like(test_preds)
            
            for j, m in pbar:
                model_pred = trained_models[m][i].predict_proba(test)
                ttest += fold_weights[j] * model_pred
                
                progress.advance(pt)

            # only one move forward
            test_preds += ttest / np.sum(fold_weights)


            progress.advance(task)

    test_preds /= n_folds

    cfg.preds = np.argmax(test_preds, axis=1)

    return np.argmax(test_preds, axis=1)



def create_submission(plot=False, cfg=cfg):
    output = pd.DataFrame()

    preds = pd.Series(cfg.preds).map({0: 'died', 1:'lived', 2:'euthanized'})
    output["id"] = cfg.id_
    output['outcome'] = preds

    output.to_csv(f"{cfg.BASE_PATH}/submission.csv", index=False)
    
    if plot:
        counts = preds.value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#9370DB', '#9400D3', '#A020F0'])
        plt.title("Test Predictions")
        plt.tight_layout()
        plt.show()