from flask import Flask, render_template, request, redirect, url_for, jsonify
from joblib import load
import pandas as pd

pipeline = load("disease_predictor.joblib")
#test = pd.read_csv('test_set.csv', header=None)

data_grid = pd.read_csv('training_data.csv')
cols_total = data_grid.columns
cols_total = cols_total[2:]

def requestResults(test):
    y_pred = pipeline.predict(test)
    return y_pred

def preProcess(symp_list):
    data_grid = pd.read_csv('training_data.csv')
    cols = data_grid.columns
    cols = cols[2:]
    data_frame = pd.DataFrame(columns=cols)
    new_row = []
    for i in range(len(cols)):
        new_row.append(0)
    temp = pd.Series(new_row, index=data_frame.columns)
    data_frame = data_frame.append(temp, ignore_index=True)
    
    for i in cols:
        if i in symp_list:
            data_frame[i][0] = 1
    X = data_frame.iloc[:,:].values
    return X
    

flask_runner = Flask(__name__)

@flask_runner.route('/disease_predictor_model', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        data = request.get_json()
        final_data = preProcess(data)
        result_data = requestResults(final_data)
        #print(result_data)
        return jsonify(result_data[0])
        
@flask_runner.route('/get_symptoms', methods=['GET'])
def get_symptoms():       
	return jsonify(list(cols_total))
        
    
flask_runner.run(debug=True, use_reloader=False)


