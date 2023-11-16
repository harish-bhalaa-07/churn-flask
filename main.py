import numpy as np
from flask import Flask,request,render_template
import pickle
import os

os.chdir('./')

app=Flask(__name__)
model=pickle.load(open('churn.pkl','rb'))


@app.route('/')
def home():
    return render_template('churn_prediction.html')  #html page here


@app.route('/predict',methods=['POST'])
def predict():
    
    #storing the input values from html form
    features=[[float(x) for x in request.form.values()]]
    final_features=np.array(features)  

    prediction=model.predict(final_features)
    prediction=prediction.tolist()
    
    if prediction == [0]:
        prediction = "You are a churn customer"
    else:
        prediction = "You are not a churn customer"

    return render_template('churn_prediction.html', prediction_value = prediction)

if __name__=="__main__":
    app.run(debug=True)
    
