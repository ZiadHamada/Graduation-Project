# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import sys
import pandas as pd

# Your API definition
app = Flask(__name__)

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if gbc:
        try:
            
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = list(gbc.predict(query))
            if prediction[0] == 1:
                result = 'You suffer from heart failure'
            else:
                result = 'You do not have heart failure' 
            return jsonify({'prediction': result})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    gbc = joblib.load("D:\\4Faculity\\APIs\\API heart\\new_heart_disease_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("D:\\4Faculity\\APIs\\API heart\\model_heart_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port) 