from xgboost import XGBRegressor
import xgboost as xgb
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import KFold
from pathlib import Path
import optuna
import shap


def corr_metric(y_true, y_pred):
    score = np.corrcoef(y_true, y_pred)[0, 1]
    return score

def mse_metric(y_true, y_pred):
    score = -mean_squared_error(y_true, y_pred)
    return score


class AutoXGBRegressor:
    
    def __init__(self, num_parallel_tree=5, gpuID=0):
        self.best_params = None
        self.model = XGBRegressor()
        self.best_trial = None
        self.ensemble = False
        self.study = None
        self.num_parallel_tree = num_parallel_tree
        self.gpuID = gpuID

    def objective(self, trial, X_train=None, y_train=None, X_val=None, y_val=None, cv=None):
        """
        Objective function for Optuna study.
        """
        params = {
            'eta': trial.suggest_float('eta',0.01,1),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 500,2000,log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 1, 100),
            'lambda': trial.suggest_float('lambda',0.5,3),
            'sampling_method': trial.suggest_categorical('sampling_method',['uniform','gradient_based']),
            'num_parallel_tree': self.num_parallel_tree,
            'tree_method': 'hist',  # Enable GPU acceleration
            'device': f'cuda:{self.gpuID}',  # Use GPU acceleration
            'objective': 'reg:squarederror',
            }

        params['eval_metric'] = 'rmse' # Evaluation metric

        if cv:
            mses_scores = []
            corr_scores = []
            
            for data in cv:
                
                X_train,y_train,X_val,y_val,*_ = data
                
                dtrain = xgb.DMatrix(X_train, y_train)
                dval = xgb.DMatrix(X_val, y_val)
                
                booster = xgb.train(params, dtrain, evals=[(dval, 'eval')], num_boost_round=params['n_estimators'], early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=False)
                booster.set_param({'device':f'cuda:{self.gpuID}'})
                
                y_pred = booster.predict(dval)
                
                mses_scores.append(mean_squared_error(y_val, y_pred))
                corr_scores.append(np.corrcoef(y_val, y_pred)[0, 1])
                
            return sum(mses_scores) / len(mses_scores), sum(corr_scores) / len(corr_scores)
        
        else:

            if X_val is not None and y_val is not None:

                dtrain = xgb.DMatrix(X_train, y_train)
                dval = xgb.DMatrix(X_val, y_val)
                
                booster = xgb.train(params, dtrain, evals=[(dval, 'eval')], num_boost_round=params['n_estimators'], early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=False)
                
                y_pred = booster.predict(dval)
                
                return mean_squared_error(y_val, y_pred), np.corrcoef(y_val, y_pred)[0, 1]

        

    def search(self, 
               X_train=None, y_train=None, 
               X_val=None, y_val=None, 
               cv=None, 
               n_trials=300, n_startup_trials=100,
               optim_seed=0):
        """
        Search the best hyperparameters using Optuna.
        """
        
        sampler = optuna.samplers.TPESampler(seed=optim_seed, n_startup_trials=n_startup_trials,multivariate=True)
        self.study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
                                    #storage=storage, study_name=drug, load_if_exists=True)

        if cv is None:
            self.study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
            self.ensemble = False

        else:
            self.study.optimize(lambda trial: self.objective(trial, cv=cv), n_trials=n_trials)
            self.ensemble = True
        

    def predict(self, X):
        if self.ensemble:
            preds = []
            for model in self.models:
                preds.append(model.predict(X))
            return np.mean(preds, axis=0)

        else:
            return self.model.predict(X)