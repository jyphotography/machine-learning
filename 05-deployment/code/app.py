from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and other necessary components
with open('/workspaces/machine-learning/05-deployment/code/churn-model.bin', 'rb') as f:
    model = pickle.load(f)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Process the data, make predictions using the model, and return the results
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)