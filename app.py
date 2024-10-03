import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def predicting(object):
    
    model = joblib.load('model.pkl')
    df = pd.DataFrame({
        'Extraversion Score':[object['Extraversion Score']], 
        'Thinking Score':[object['Thinking Score']],
        'Age':[object['Age']],
        'Sensing Score':[object['Sensing Score']],
        'Judging Score':[object['Judging Score']]
        })
    return model.predict(df)

@app.route('/predict', methods=['POST'])
def processing():
    try:
        data = request.get_json()
        required_fields = {
            'extraversion_score': 'Extraversion Score',
            'thinking_score': 'Thinking Score',
            'age': 'Age',
            'sensing_score': 'Sensing Score',
            'judging_score': 'Judging Score'
        }

        if not all(field in data for field in required_fields.keys()):
            return jsonify({'error': 'Thiếu dữ liệu yêu cầu'}), 400

        transformed_data = {required_fields[key]: value for key, value in data.items()}

        prediction = predicting(transformed_data)
        return jsonify({'data': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


