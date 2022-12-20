import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    show = ''

    if output == 0.0:
        show = show + "not a right day to invest."
    else:
        show = show + "right day to invest."

    return render_template('index.html', prediction_text='Today is {}'.format(show))


if __name__ == "__main__":
    app.run(debug=True)