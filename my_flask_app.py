from flask import Flask, render_template, request, jsonify
import joblib



app = Flask(__name__)
#@app.route('/')
def index():
    title = "Machine Failure"
    return render_template('index.html', title=title)

# Load the serialized model
import pickle

# Load the serialized model
#loaded_model = pickle.load(open('model.pkl', 'rb'))

def rule_based_classifier_with_conditions(features):
    conditions = {
    "TWF": lambda x: x["Tool_wear"] >= 200 and x["Tool_wear"] <= 240,
    "HDF": lambda x: abs(x["Process_temperature"] - x["Air_temperature"]) < 8.6 and x["Rotational_speed"] < 1380,
    "PWF": lambda x: (((x["Torque"] * x["Rotational_speed"]) * 2 * 3.141592653589793) / 60) < 3500 or (((x["Torque"] * x["Rotational_speed"])* 2 * 3.141592653589793 ) / 60) > 9000,
    "OSF": lambda x: (x["Type"] == 1 and (x["Tool_wear"] * x["Torque"]) > 11000) or (x["Type"] == 2 and (x["Tool_wear"] * x["Torque"]) > 12000) or (x["Type"] == 3 and (x["Tool_wear"] * x["Torque"]) > 13000),
    #"Machine_failure": lambda x: x["TWF"] == 1 or x["OSF"]==1 or x["PWF"]==1 or x["HDF"]==1
    }
    classes = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    #"Machine_failure": "Machine_failure"
    }
    detected_failures = []  

    
    for failure_mode, condition in conditions.items():
        if condition(features):
            detected_failures.append(classes[failure_mode])

   
    if detected_failures:
        return detected_failures
    else:
        return ['No failure detected']
loaded_model = rule_based_classifier_with_conditions
'''
@app.route('/')
def index():
    title = "Machine Failure"
    return render_template('prediction.html', title=title)
'''
sample_features = {
    "Tool_wear": 220,
    "Process_temperature": 200,
    "Air_temperature": 190,
    "Rotational_speed": 1200,
    "Torque": 30,
    "Type": 2
}
@app.route('/', methods=['GET'])
def predict():
    # Get the input data from the request
    #data = request.get_json()
    data = sample_features
    # Perform prediction using the loaded model
    prediction = loaded_model(data)[0]
    title = "Machine Failure"
    # Return the prediction as JSON response
    return render_template('prediction.html', prediction=prediction, title=title)#jsonify({'prediction': prediction.tolist()})
if __name__ == "__main__":
    app.run(debug=True)