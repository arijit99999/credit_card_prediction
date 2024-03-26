from src.CreditCardDefaultPrediction.pipelines.prediction_pipeline import custom_data,model_prediction
from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__,template_folder="template")
@app.route('/')
def home_page():
    return render_template("form.html")
@app.route('/predict',methods=["POST"])
def pred_page():   
        get_data=custom_data(LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX= float(request.form.get('SEX')),
            EDUCATION= float(request.form.get('EDUCATION')),
            MARRIAGE= float(request.form.get('MARRIAGE')),
            AGE= float(request.form.get('AGE')),
            PAY_0= float(request.form.get('PAY_0')),
            PAY_2= float(request.form.get('PAY_2')),
            PAY_3= float(request.form.get('PAY_3')),
            PAY_4= float(request.form.get('PAY_4')),
            PAY_5= float(request.form.get('PAY_5')),
            PAY_6= float(request.form.get('PAY_6')),
            BILL_AMT1= float(request.form.get('BILL_AMT1')),
            BILL_AMT6= float(request.form.get('BILL_AMT6')),
            PAY_AMT1=float(request.form.get('PAY_AMT1')),
            PAY_AMT2=float(request.form.get('PAY_AMT2')),
            PAY_AMT3=float(request.form.get('PAY_AMT3')),
            PAY_AMT4=float(request.form.get('PAY_AMT4')),
            PAY_AMT5=float(request.form.get('PAY_AMT5')),
            PAY_AMT6=float(request.form.get('PAY_AMT6')))
        
        final_data=get_data.get_data_as_dataframe()
        pred=model_prediction()
        x=pred.model_pred_initiate(final_data)
        x=x[0]
        x=np.where(x>0.0,"yes","no")
        return render_template("result.html",x=x)

if __name__=="__main__":
 app.run(host="0.0.0.0", port = 8080)
