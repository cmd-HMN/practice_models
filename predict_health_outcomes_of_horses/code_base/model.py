from configs import cfg
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb

class Classifier:
    def __init__(self, model_type='base', cfg=cfg):
        super().__init__()
        self.model_type = model_type 
        self.models = self.prepare_model()
        self.len_models = len(self.models)


    def prepare_model(self):
        if self.model_type == 'base':
            return [SVC(decision_function_shape=cfg.decision_function_shape)]

        elif self.model_type == 'main':
            xgb_params = {
                    'n_estimators': cfg.n_estimators,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'subsample': 0.8,
                    'colsample_bytree': 0.1,
                    'n_jobs': -1,
                    'eval_metric': 'merror',
                    'objective': 'multi:softmax',
                    'tree_method': 'hist',
                    'verbosity': 0,
                    'random_state': cfg.random_state,
                    'class_weight':cfg.class_weights_dict,
                }
            if cfg.device == 'gpu':
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['predictor'] = 'gpu_predictor'
                    
            xgb_params2=xgb_params.copy() 
            xgb_params2['subsample']= 0.3
            xgb_params2['max_depth']=8
            xgb_params2['learning_rate']=0.005
            xgb_params2['colsample_bytree']=0.9
    
            
            lgb_params = {
                'n_estimators': cfg.n_estimators,
                'max_depth': 8,
                'learning_rate': 0.02,
                'subsample': 0.20,
                'colsample_bytree': 0.56,
                'reg_alpha': 0.25,
                'reg_lambda': 5e-08,
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'device': cfg.device,
                'random_state': cfg.random_state,
                'class_weight':cfg.class_weights_dict,
                'verbosity': -1
            }
            lgb_params2 = {
                'n_estimators': cfg.n_estimators,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.20,
                'colsample_bytree': 0.56,
                'reg_alpha': 0.25,
                'reg_lambda': 5e-08,
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'device': cfg.device,
                'random_state': cfg.random_state,
                'class_weight':cfg.class_weights_dict,
                'verbosity': -1
            }
            lgb_params3=lgb_params.copy()  
            lgb_params3['subsample']=0.9
            lgb_params3['reg_lambda']=0.3461495211744402
            lgb_params3['reg_alpha']=0.3095626288582237
            lgb_params3['max_depth']=9
            lgb_params3['learning_rate']=0.007
            lgb_params3['colsample_bytree']=0.5
    
                    
            cb_params = {
                'iterations': cfg.n_estimators,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 0.7,
                'random_strength': 0.2,
                'max_bin': 200,
                'od_wait': 65,
                'one_hot_max_size': 70,
                'grow_policy': 'Depthwise',
                'bootstrap_type': 'Bayesian',
                'od_type': 'Iter',
                'eval_metric': 'TotalF1',
                'loss_function': 'MultiClass',
                'task_type': cfg.device.upper(),
                'random_state': cfg.random_state,
            }
            cb_sym_params = cb_params.copy()
            cb_sym_params['grow_policy'] = 'SymmetricTree'
            cb_loss_params = cb_params.copy()
            cb_loss_params['grow_policy'] = 'Lossguide'
            
            cb_params2=  cb_params.copy()
            cb_params2['learning_rate']=0.01
            cb_params2['depth']=8
            
            cb_params3={
                'iterations': cfg.n_estimators,
                'random_strength': 0.1, 
                'one_hot_max_size': 70, 'max_bin': 100, 
                'learning_rate': 0.008, 
                'l2_leaf_reg': 0.3, 
                'grow_policy': 'Depthwise', 
                'depth': 9, 
                'max_bin': 200,
                'od_wait': 65,
                'bootstrap_type': 'Bayesian',
                'od_type': 'Iter',
                'eval_metric': 'TotalF1',
                'loss_function': 'MultiClass',
                'task_type': cfg.device.upper(),
                'random_state': cfg.random_state,
            }
            models = {
                'svc': SVC(gamma="auto", probability=True, random_state=cfg.random_state),
                'xgb': xgb.XGBClassifier(**xgb_params),
                'xgb2': xgb.XGBClassifier(**xgb_params2),
                'lgb': lgb.LGBMClassifier(**lgb_params),
                'lgb2': lgb.LGBMClassifier(**lgb_params2),
                'lgb3': lgb.LGBMClassifier(**lgb_params3),
                'cat': CatBoostClassifier(**cb_params),
                "cat_sym": CatBoostClassifier(**cb_sym_params),
                "cat_loss": CatBoostClassifier(**cb_loss_params),
                'cat2': CatBoostClassifier(**cb_params2),
                'rf': RandomForestClassifier(n_estimators=1000, random_state=cfg.random_state),
                'hist_gbm' : HistGradientBoostingClassifier (max_iter=300, learning_rate=0.001,  max_leaf_nodes=80,
                                                             max_depth=6,class_weight=cfg.class_weights_dict, random_state=cfg.random_state)
            }
            return models

        else:
            print(f"No such choice {self.model_type} --- valid are [main/base]")