import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas

app = Flask(__name__)

model = pickle.load(open('cal_housing_model.pkl','rb'))
scaler = pickle.load(open('cal_housing_preprocessing_scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    #this makes sure input is always in json from the api
    data=request.json['data']
    print(data)
    data=np.array(list(data.values())).reshape(1,-1)
    data=scaler.transform(data)
    output=model.predict(data)[0]
    print(output)
    return jsonify(output)

@app.route('/predict',methods=['POST'])

def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template('home.html',prediction_text='The predicted median house price (in $100,00) is {}'.format(output))


if __name__=='__main__':
    app.run(debug='True')