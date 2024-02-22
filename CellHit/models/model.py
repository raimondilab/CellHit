from xgboost import XGBRegressor
import numpy as np
import shap
import xgboost as xgb

class CustomXGBoost():

    def __init__(self, base_params):
        """
        Initialize the CustomXGBoost class.
        
        Parameters:
        params (dict): Parameters to be passed to the XGBRegressor model.
        """
        self.base_params = base_params

    def fit(self, train_X, train_Y, valid_X, valid_Y):

        dtrain = xgb.DMatrix(train_X, train_Y)
        dval = xgb.DMatrix(valid_X, valid_Y)

        self.model = xgb.train(self.base_params, dtrain, evals=[(dval, 'eval')], num_boost_round=self.base_params['n_estimators'], early_stopping_rounds=self.base_params['early_stopping_rounds'], verbose_eval=False)

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
    def __init__(self, base_params):
        """
        Initialize the EnsembleXGBoost class.
        
        Parameters:
        base_params (dict): Parameters to be passed to each XGBRegressor model.
        """
        self.models = []
        self.base_params = base_params

    def fit(self, data_subset):
        """
        Fit multiple XGBRegressor models based on the training-validation splits.
        
        Parameters:
        data_subset (list): List of tuples containing training, validation and test data.
        """
        for data in data_subset:
            
            dtrain = xgb.DMatrix(data['train_X'], data['train_Y'])
            dval = xgb.DMatrix(data['valid_X'], data['valid_Y'])
                
            booster = xgb.train(self.base_params, dtrain, evals=[(dval, 'eval')], num_boost_round=self.base_params['n_estimators'], early_stopping_rounds=self.base_params['early_stopping_rounds'], verbose_eval=False)
                
            self.models.append(booster)

            #model = XGBRegressor(**self.base_params)
            #model.fit(X_train.values, y_train, eval_set=[(X_valid.values, y_valid)], verbose=False)
            #self.models.append(model)
            

    def predict(self,test_X,return_shaps=False):
        """
        Make predictions based on the ensemble of models and average them.
        
        Parameters:
        X_test (DataFrame): Test features.
        
        Returns:
        np.array: Averaged predictions.
        """
        preds = []
        shaps = []
        
        for model in self.models:
            #check whethere test_
            dtest = xgb.DMatrix(test_X)
            preds.append(model.predict(dtest))
            
            if return_shaps:
                explainer = shap.TreeExplainer(model)
                #shaps.append(explainer.shap_values(X_test.values))
                shaps.append(explainer(test_X.values))
                
        if return_shaps:
            #obtain a tridimensional array with the shap values
            shap_values = np.array([x.values for x in shaps])
            shap_values = np.mean(shap_values,axis=0)
            shap_base_values = np.array([x.base_values for x in shaps])
            shap_base_values = np.mean(shap_base_values,axis=0)
            #obtain the mean of the shap values
            
            #if inverse_map:
            #    feature_names = [inverse_map[x] for x in X_test.columns]
            #else:   
            feature_names = test_X.columns
           
            explanation = shap.Explanation(values=shap_values,
                                base_values=shap_base_values,
                                data=test_X.values,
                                feature_names=feature_names)
            
            #return np.mean(preds, axis=0),explanation
            return {'predictions':np.mean(preds, axis=0), 'shap_values':explanation}
        
        return {'predictions':np.mean(preds, axis=0)}
    
    def get_important_features(self):
        return np.mean([model.feature_importances_ for model in self.models], axis=0)
    
    