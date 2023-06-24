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
                result = 'You suffer from \"Allergy\"'
            elif prediction[0] == 1:
                result = 'You suffer from \"Common Cold\"' 
            elif prediction[0] == 2:
                result = 'You suffer from \"Acne\"'
            elif prediction[0] == 3:
                result = 'You suffer from \"Arthritis\"'
            elif prediction[0] == 4:
                result = 'You suffer from \"Osteoarthristis\"'
            elif prediction[0] == 5:
                result = 'You suffer from \"Hypoglycemia\"'
            elif prediction[0] == 6:
                result = 'You suffer from \"Hyperthyroidism\"'
            elif prediction[0] == 7:
                result = 'You suffer from \"Hypothyroidism\"' 
            elif prediction[0] == 8:
                result = 'You suffer from \"Varicose veins\"'
            elif prediction[0] == 9:
                result = 'You suffer from \"Heart attack\"' 
            elif prediction[0] == 10:
                result = 'You suffer from \"Pneumonia\"'
            elif prediction[0] == 11:
                result = 'You suffer from \"Hepatitis C\"' 
            elif prediction[0] == 12:
                result = 'You suffer from \"GERD\"'
            elif prediction[0] == 13:
                result = 'You suffer from \"Hepatitis B\"'
            elif prediction[0] == 14:
                result = 'You suffer from \"Typhoid\"'
            elif prediction[0] == 15:
                result = 'You suffer from \"Paralysis \(brain hemorrhage\)\"'
            elif prediction[0] == 16:
                 result = 'You suffer from \"Migraine\"'
            elif prediction[0] == 17:
                 result = 'You suffer from \"Hypertension\"'
            elif prediction[0] == 18:
                 result = 'You suffer from \"Gastroenteritis\"'
            elif prediction[0] == 19:
                 result = 'You suffer from \"Diabetes\"'
            elif prediction[0] == 20:
                 result = 'You suffer from \"Peptic ulcer diseae\"'
            elif prediction[0] == 21:
                 result = 'You suffer from \"Drug Reaction\"'
            elif prediction[0] == 22:
                result = 'You suffer from \"Urinary tract infection\"'
            
            
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
        port = 45678 # If you don't provide any port the port will be set to 12345

    gbc = joblib.load("general_diseases_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_general_diseases_columnss.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port) 