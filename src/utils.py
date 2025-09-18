from src.exceptions import CustomException
import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'wb') as file:
             
             pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        model_trained={}
        for i in range(len(list(models.keys()))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs=GridSearchCV(model,param)
            gs.fit(X_train,y_train)

            trained_model=gs.best_estimator_
            model_trained[list(models.keys())[i]]=trained_model

            y_pred_train=trained_model.predict(X_train)
            y_pred_test=trained_model.predict(X_test)

            score=r2_score(y_test,y_pred_test)
            report[list(models.keys())[i]]=score
            

        return report ,model_trained
    except Exception as e:
        raise CustomException(e,sys)

