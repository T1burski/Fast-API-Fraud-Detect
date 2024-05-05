# Fast-API-Fraud-Detect

## 1) The Model
The ML model deployed through this API was developed and explained in the Fraudulent-Transaction-Classifier (https://github.com/T1burski/Fraudulent-Transaction-Classifier). The model, in short, predicts if a financial transaction is fraudulent or not based on a selection of features regarding the transaction.

## 2) Building the API and Deploying It
The REST API was built using Fast API and tested in a server using uvicorn, a ASGI web server that provides an interface between a web server and python.

Running the web server on any IP and choosing the port 3000, we can test the API on http://127.0.0.1:3000/docs#/Predict_Fraud/predict_predict_post, which return the following screen when opened on the browser:

![image](https://github.com/T1burski/Fast-API-Fraud-Detect/assets/100734219/0989804b-f290-4e60-bbdf-6d0a718d2f7e)

When selecting the POST section in which we can test the POST method of the API, we can Try it Out by executing a request body according to the image below:

![image](https://github.com/T1burski/Fast-API-Fraud-Detect/assets/100734219/43fdbbd7-925c-4826-a48b-b909c5e584e0)

Then, by Executing it, the response body shows the results returned:

![image](https://github.com/T1burski/Fast-API-Fraud-Detect/assets/100734219/77616ed3-6169-4167-9ea8-51c7bd52971c)

For this example, we can see that the ML API returned, for the transaction characterized by the request body, the information that the transaction is a potential fraud associated with a probability of 56% that it is a fraud.
