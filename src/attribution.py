import numpy as np

def explain_anomalies(scores):
    threshold = np.percentile(scores, 95)
    anomalies = np.where(scores > threshold)[0]
    print(f"Detected {len(anomalies)} anomalies at indices:", anomalies)
