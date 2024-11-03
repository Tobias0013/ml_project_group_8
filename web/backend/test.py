from joblib import load
import pandas as pd

features = {
    "cpu_allocation_efficiency": 0.597308986,
    "memory_allocation_efficiency": 0.35329184,
    "disk_io": 666.0098057,
    "node_cpu_usage": 55.13999613,
    "node_memory_usage": 32.3140218,
    "cpu_request": 1.5151272156149198,
    "cpu_limit": 0.5366306723940306,
    "memory_request": 1183.4286644853166,
    "memory_limit": 6163.6197223284735,
    "cpu_usage": 0.9687976425514012,
    "memory_usage": 2698.4013693655243,
    "restart_count": 1,
    "uptime_seconds": 40527,
    "network_bandwidth_usage": 786.4419971231081
}

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

model = load("backend/model/model.pkl")
scaler = load("backend/model/scaler.pkl")
cluster = load("backend/model/cluster.pkl")


data_scaled = scaler.transform(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

cluster_pred = cluster.predict(data)

data["KMeans_Cluster"] = cluster_pred

y_pred = model.predict(data)


print(y_pred)