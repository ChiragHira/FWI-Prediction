import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import lasso regressor model and standard scaler pickle file
with open("model/lasso.pkl",'rb') as f1:
   lasso_model = pickle.load(f1)
 
with open("model/scaler.pkl",'rb') as f2:
   standard_scaler = pickle.load(f2)
   
   
## Route for home page
@app.route('/') # type: ignore
def index():
    return render_template("index.html")

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    if request.method == 'POST':
        Temprature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("WS"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        cla = float(request.form.get("Cla"))
        region = float(request.form.get("Region"))
        
        new_data_scaled = standard_scaler.transform([[Temprature,RH,WS,Rain,FFMC,DMC,ISI,cla,region]])
        result = lasso_model.predict(new_data_scaled)
        
        return render_template("index.html",Result=result[0])
        
    else:
        return render_template ("index.html")
        
    
    

if __name__=="__main__":
    app.run(host="0.0.0.0")