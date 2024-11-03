from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__, template_folder="static")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['cpu_allocation_efficiency'],
        data['memory_allocation_efficiency'],
        data['disk_io'],
        data["node_cpu_usage"],
        data["node_memory_usage"],
        data['cpu_request'],
        data['cpu_limit'],
        data['memory_request'],
        data['memory_limit'],
        data['cpu_usage'],
        data['memory_usage'],
        data['restart_count'],
        data['uptime_seconds'],
        data['network_bandwidth_usage']
    ]

    # Convert features 
    data = pd.DataFrame([features], columns=[
        "cpu_allocation_efficiency",
        "memory_allocation_efficiency",
        "disk_io",
        "node_cpu_usage",
        "node_memory_usage",
        "cpu_request",
        "cpu_limit",
        "memory_request",
        "memory_limit",
        "cpu_usage",
        "memory_usage",
        "restart_count",
        "uptime_seconds",
        "network_bandwidth_usage"
    ])

    # Loads our model, scaler and cluster
    model = load("backend/model/model.pkl")
    scaler = load("backend/model/scaler.pkl")
    cluster = load("backend/model/cluster.pkl")

    # Scale data and convert back to dataframe
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    # Predict cluster
    cluster_pred = cluster.predict(data)
    data["KMeans_Cluster"] = cluster_pred

    # Make prediction with model
    y_pred = model.predict(data)

    return jsonify({'prediction': float(y_pred)})
    
if __name__ == '__main__':
    app.run(debug=True)


