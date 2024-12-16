from xgboost import XGBRegressor
import numpy as np
import shap
import xgboost as xgb
import os

import tempfile
import shutil

class CustomXGBoost():

    def __init__(self, base_params):
        """
        Initialize the CustomXGBoost class.
        
        Parameters:
        params (dict): Parameters to be passed to the XGBRegressor model.
        """
        self.base_params = base_params

    def fit(self, train_X, train_Y, valid_X, valid_Y, fix_seed=False):

        dtrain = xgb.DMatrix(train_X, train_Y)
        dval = xgb.DMatrix(valid_X, valid_Y)

        train_params = {k: v for k, v in self.base_params.items() if k not in ['n_estimators', 'early_stopping_rounds']} #used to get rid of the warnings when training the model
        self.model = xgb.train(train_params, dtrain, evals=[(dval, 'eval')], num_boost_round=self.base_params['n_estimators'], early_stopping_rounds=self.base_params['early_stopping_rounds'], verbose_eval=False)

    def predict(self, test_X, return_shaps=False):
        
        dtest = xgb.DMatrix(test_X)
        out = {}
        out['predictions'] = self.model.predict(dtest)
        
        if return_shaps:
            explainer = shap.TreeExplainer(self.model)
            shaps = explainer(test_X.values)
            return {**out, **{'shap_values':shaps}}
        
        return out
    
    def get_important_features(self):
        return self.model.get_score(importance_type='gain')


class EnsembleXGBoost():
    def __init__(self, base_params=None):
        """
        Initialize the EnsembleXGBoost class.
        
        Parameters:
        base_params (dict): Parameters to be passed to each XGBRegressor model.
        """
        self.models = []
        self.base_params = base_params

    def fit(self, data_subset,fix_seed=False):
        """
        Fit multiple XGBRegressor models based on the training-validation splits.
        
        Parameters:
        data_subset (list): List of tuples containing training, validation and test data.
        """
        for idx,data in enumerate(data_subset):
            
            dtrain = xgb.DMatrix(data['train_X'], data['train_Y'])
            dval = xgb.DMatrix(data['valid_X'], data['valid_Y'])

            if fix_seed:
                self.base_params['seed'] = idx

            train_params = {k: v for k, v in self.base_params.items() if k not in ['n_estimators', 'early_stopping_rounds']} #used to get rid of the warnings when training the model
            booster = xgb.train(train_params, dtrain, evals=[(dval, 'eval')], 
                                num_boost_round=self.base_params['n_estimators'], 
                                early_stopping_rounds=self.base_params['early_stopping_rounds'], 
                                verbose_eval=False)
                
            self.models.append(booster)
            

    def predict(self,test_X,return_shaps=False,return_stds=False):
        """
        Make predictions based on the ensemble of models and average them.
        
        Parameters:
        X_test (DataFrame): Test features.
        
        Returns:
        np.array: Averaged predictions.
        """
        preds = []
        shaps = []

        feature_names = self.models[0].feature_names

        #check whethere test_X has the same features as the model
        try:
            test_X = test_X[feature_names]
        except:
            raise ValueError(f'Test data has different features than the model. Check the feature names.')

        dtest = xgb.DMatrix(test_X)
        
        for model in self.models:
            
            preds.append(model.predict(dtest))
            
            if return_shaps:
                explainer = shap.TreeExplainer(model)
                shaps.append(explainer(test_X.values))
                
        if return_shaps:
            #obtain a tridimensional array with the shap values
            shap_values = np.array([x.values for x in shaps])
            shap_values = np.mean(shap_values,axis=0)
            shap_base_values = np.array([x.base_values for x in shaps])
            shap_base_values = np.mean(shap_base_values,axis=0)
            
            feature_names = test_X.columns
           
            explanation = shap.Explanation(values=shap_values,
                                base_values=shap_base_values,
                                data=test_X.values,
                                feature_names=feature_names,
                                instance_names=list(test_X.index))
        
        output = {}
        output['predictions'] = np.mean(preds, axis=0)

        if return_shaps:
            output['shap_values'] = explanation

        if return_stds:
            output['std'] = np.std(preds, axis=0)
        
        return output
    
    def get_important_features(self):
        return np.mean([model.feature_importances_ for model in self.models], axis=0)
    

    def save_model(self, path):
        # Create a temporary directory to save individual models
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_file:
            # Write number of models
            np.save(temp_file, len(self.models))
            
            # Save each model's binary data
            for model in self.models:
                # Get model binary data
                model_data = model.save_raw()
                # Save length and data
                np.save(temp_file, len(model_data))
                temp_file.write(model_data)
                
        # Move temporary file to final destination
        shutil.move(temp_file.name, path)

    @classmethod
    def load_model(cls, path,limit_load:int=None):
        instance = cls(None)
        instance.models = []
        
        with open(path, 'rb') as f:
            # Read number of models
            n_models = np.load(f)

            if limit_load is not None:
                n_models = limit_load
            else:
                n_models = int(n_models)
            
            for _ in range(n_models):
                # Read model size
                model_size = np.load(f)
                # Read model data
                model_data = f.read(int(model_size))
                
                # Create new booster
                model = xgb.Booster()
                model.load_model(bytearray(model_data))
                instance.models.append(model)
            
        return instance
    

    @classmethod
    def legacy_load_model(cls, path):
        instance = cls(None)
        instance.models = []
        i = 0
        while True:
            model_path = f'{path}/{i}.json'
            if not os.path.exists(model_path):
                break
                #raise ValueError(f'Model {model_path} does not exist')
            model = xgb.Booster()
            model.load_model(model_path)
            instance.models.append(model)
            i += 1

        if len(instance.models) == 0:
            raise ValueError(f'No models found in {path}')
        
        return instance