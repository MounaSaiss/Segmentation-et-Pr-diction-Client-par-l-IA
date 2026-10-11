import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


#function to calculate the total amount spent by a customer
def calculate_total_amount(quantity, unit_price):
    return quantity * unit_price


#function to calculate recency, frequency, and monetary value (RFM) for each customer
def calculate_rfm(df_clean):
    reference_date = df_clean["InvoiceDate"].max()

    rfm = df_clean.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique")
    )

    rfm["Recency"] = (
        reference_date - rfm["LastPurchase"]
    ).dt.days

    df = df_clean.copy()
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    rfm["Monetary"] = df.groupby("CustomerID")["TotalAmount"].sum()

    return rfm[["Recency", "Frequency", "Monetary"]]


#function calcule transformation log et standardisation des données RFM
def preprocess_rfm(rfm):
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].copy()

    rfm_log = np.log1p(rfm_log)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    rfm_scaled = pd.DataFrame(
        rfm_scaled,
        columns=["Recency", "Frequency", "Monetary"],
        index=rfm.index
    )

    return rfm_log, rfm_scaled, scaler


#function to train a KMeans model on the scaled RFM data
def train_kmeans(rfm_scaled):
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(rfm_scaled)

    return model, clusters

#function to predict the cluster for a new client using the trained model
def predict_new_client(model, new_client):
    return model.predict(new_client)



#functions to save and load the trained model using joblib
def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
