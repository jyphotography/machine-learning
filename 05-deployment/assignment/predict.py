from flask import Flask
from flask import request
from flask import jsonify
import pickle

model_file = "model1.bin"
with open(model_file, 'rb')as f_in:
    model = pickle.load(f_in)
dv_file = "dv.bin"
with open(dv_file, 'rb')as f_in:
    dv = pickle.load(f_in)

app = Flask('predict')
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    result = {
        'credit_probability':round(y_pred, 3)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)