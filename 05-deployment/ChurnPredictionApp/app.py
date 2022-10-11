import pickle
from flask import Flask, request, jsonify

logistic_model_binary = './models/logistic_classifier.bin'
dict_vectorizer_binary = './models/dict_vectorizer.bin'

with open(logistic_model_binary, "rb") as logistic_model_file:
    logistic_model = pickle.load(logistic_model_file)

with open(dict_vectorizer_binary, "rb") as dict_vectorizer_file:
    dict_vectorizer = pickle.load(dict_vectorizer_file)

app = Flask('churn_prediction_app')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dict_vectorizer.transform([customer])
    y_pred = logistic_model.predict_proba(X)[0, 1]
    card = y_pred >= 0.5

    result = {
        'probability': float(y_pred),
        'card': bool(card)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
