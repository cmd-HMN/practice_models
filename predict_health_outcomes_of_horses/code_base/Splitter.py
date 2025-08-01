from sklearn.model_selection import KFold, train_test_split
from configs import cfg


class Splitter:
    def __init__(self, cfg=cfg, kfold=True):
        super().__init__()
        self.test_size = cfg.test_size
        self.kfold = kfold
        self.n_splits = cfg.n_splits

    def split_data(self, X, y):
        if self.kfold:
            kf = KFold(n_splits=self.n_splits, random_state=cfg.random_state, shuffle=True)
            for train_index, val_index in kf.split(X, y):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                yield X_train, X_val, y_train, y_val
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
            yield X_train, X_val, y_train, y_val