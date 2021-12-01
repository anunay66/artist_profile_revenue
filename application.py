
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


application = Flask(__name__)



application.secret = os.environ.get('SECRET')



model = pickle.load(open('modell.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('ind.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('ind.html', prediction_text='should be around $ {}'.format(output))


if __name__ == "__main__":
    application.run(debug=True)
