from sklearn.metrics import f1_score
from configs import cfg


def score(y_true, y_pred, cfg=cfg):
    return f1_score(y_true, y_pred, average="micro")