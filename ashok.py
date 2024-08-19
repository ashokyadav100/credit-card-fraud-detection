from flask import Flask, request, jsonify
import pickle
import numpy as np  # Assuming you're using NumPy for data manipulation

app = Flask(__name__)

# Save the model and scaler (assuming you have these defined)
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['CreditScore'], data['Location'], data['Gender'],
                         data['Age'], data['Tenure'], data['AccountBalance'],
                         data['NumOfProducts'], data['HasCrCard'],
                         data['IsActiveMember'], data['EstimatedSalary']])

    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)

    return jsonify({'Exited': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)