from sklearn.ensemble import RandomForestClassifier
import numpy as np

'''
module created to store the algorithm chosen to be used
and the prediction function based on the classification
threshold pre-defined in the EDA
'''

class ModelFunctions:
    def __init__(self):
        self.model = RandomForestClassifier(class_weight='balanced')
    
    def train_model(self, X, y):
        '''
        does the model fit (train), returning the trained model
        '''
        return self.model.fit(X, y)
    
    def class_threshold(self, clf,  X_pred, threshold = 0.43):

        '''
        classifies a transaction based on the threshold of
        the predicted probability to be considered a fraud
        '''

        y_proba = clf.predict_proba(X_pred)[:, 1]
        y_pred = np.where(y_proba >= threshold, 1, 0)

        return y_pred, y_proba
