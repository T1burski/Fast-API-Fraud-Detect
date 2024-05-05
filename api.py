import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from src.model_functions import ModelFunctions
import json
import numpy as np
import pandas as pd



# setting the fast api characteristics
tags_metadata = [{"name": "fraud-detect", "description": "Fraudulent Transaction Detection API"}]
app = FastAPI(title = "Fraud Detection API",
              description = "Fraudulent Transaction Detection API",
              version = "1.0",
              contact = {"name": "Artur"},
              openapi_tags = tags_metadata)


#setting the pydantic (BaseModel) class for data validation
class Features(BaseModel):
    V4: float
    V10: float
    V12: float
    V14: float
    V20: float


#loading the model in pickle format
def load_model():
    with open('artifacts\model.pkl', 'rb') as model:
        clf_model = pickle.load(model)
    return clf_model

clf_model = load_model()


# setting get method for root
@app.get("/")
def message():
    text = "API for fraudulent transactions detection in the financial institution"
    return text


# setting post method - predict using the model
@app.post("/predict", tags = ["Predict_Fraud"])
async def predict(Features: Features):

    #receives json and converts to python dict
    X_data = Features.model_dump_json()
    X_data_dict = json.loads(X_data)
    
    # sets up features from dict
    V4 = X_data_dict["V4"]
    V10 = X_data_dict["V10"]
    V12 = X_data_dict["V12"]
    V14 = X_data_dict["V14"]
    V20 = X_data_dict["V20"]

    # creates input structure to match the format that the model was trained
    features = [V4, V10, V12, V14, V20]
    input = pd.DataFrame(np.array(features).reshape(1,-1), columns=["V4", "V10", "V12", "V14", "V20"])

    # predicts, returning the class and proba of fraud
    output = ModelFunctions().class_threshold(clf=clf_model, X_pred=input)[0][0]
    output_prob = ModelFunctions().class_threshold(clf=clf_model, X_pred=input)[1][0]

    if output == 1:
        output_text = "Yes"
    else:
        output_text = "No"

    # builds return dict
    response = {"Potential Fraud": output_text,
                "Probability to be Fraud": round(output_prob*100, 2)}
    
    return response

# running the api using uvicorn web server interface
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3000)