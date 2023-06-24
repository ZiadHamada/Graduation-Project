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
            if prediction[0] == 0:
                result = 'Type of Migraine is \"Typical aura with migraine\"'
            elif prediction[0] == 1:
                result = 'Type of Migraine is \"Migraine without aura\"' 
            elif prediction[0] == 2:
                result = 'Type of Migraine is \"Familial hemiplegic migraine\"' 
            elif prediction[0] == 3:
                result = 'Type of Migraine is \"Typical aura without migraine\"' 
            elif prediction[0] == 4:
                result = 'Type of Migraine is \"Basilar-type aura\"' 
            elif prediction[0] == 5:
                result = 'Type of Migraine is Unknown' 
            elif prediction[0] == 6:
                result = 'Type of Migraine is \"Sporadic hemiplegic migraine\"' 
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
        port = 34567 # If you don't provide any port the port will be set to 12345

    gbc = joblib.load("migraine_disease_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_migraine_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port) 