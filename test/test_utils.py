import pandas as pd
import numpy as np
from utils import preprocess_rfm
from utils import calculate_rfm
from utils import train_kmeans
from utils import calculate_total_amount
from utils import predict_new_client
from sklearn.ensemble import RandomForestClassifier
from utils import save_model, load_model
from sklearn.ensemble import RandomForestClassifier

#data  frame de test pour la fonction calculate_total_amount
def test_calculate_total_amount():
    quantity = pd.Series([2, 3, 5])
    unit_price = pd.Series([10, 20, 4])

    result = calculate_total_amount(quantity, unit_price)

    assert result.tolist() == [20, 60, 20]
    
    

#data frame de test pour la fonction calculate_rfm
def test_calculate_rfm():
    df = pd.DataFrame({
        "CustomerID": [101, 101, 102],
        "InvoiceNo": ["A001", "A002", "B001"],
        "InvoiceDate": pd.to_datetime([
            "2026-01-01",
            "2026-01-05",
            "2026-01-03"
        ]),
        "Quantity": [2, 3, 1],
        "UnitPrice": [10, 10, 20]
    })

    rfm = calculate_rfm(df)

    assert rfm.loc[101, "Recency"] == 0
    assert rfm.loc[101, "Frequency"] == 2
    assert rfm.loc[101, "Monetary"] == 50

    assert rfm.loc[102, "Recency"] == 2
    assert rfm.loc[102, "Frequency"] == 1
    assert rfm.loc[102, "Monetary"] == 20


#data frame de test pour la fonction preprocess_rfm
def test_preprocess_rfm():
    rfm = pd.DataFrame({
        "Recency": [1, 10, 30],
        "Frequency": [1, 5, 10],
        "Monetary": [100, 500, 1000]
    })

    rfm_log, rfm_scaled, scaler = preprocess_rfm(rfm)

    assert rfm_log.shape == (3, 3)
    assert rfm_scaled.shape == (3, 3)
    assert np.isfinite(rfm_scaled.to_numpy()).all()
    assert np.allclose(rfm_scaled.mean().to_numpy(), 0)
    assert scaler is not None
    
    

def test_train_kmeans():
    rfm_scaled = np.array([
        [-2.0, -2.0, -2.0],
        [-1.8, -2.1, -1.9],
        [0.0, 0.0, 0.0],
        [0.2, -0.1, 0.1],
        [2.0, 2.0, 2.0],
        [2.1, 1.9, 2.2]
    ])

    model, clusters = train_kmeans(rfm_scaled)

    assert len(clusters) == len(rfm_scaled)
    assert len(np.unique(clusters)) == 3
    assert model.n_clusters == 3
    
    

#data frame de test pour la fonction predict_new_client
def test_predict_new_client():
    X_train = pd.DataFrame({
        "Recency": [10, 20, 100, 150, 30, 200],
        "Frequency": [12, 10, 2, 1, 8, 1],
        "Monetary": [5000, 4000, 300, 200, 3000, 100]
    })

    y_train = [0, 0, 1, 1, 0, 2]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    new_client = pd.DataFrame({
        "Recency": [10],
        "Frequency": [12],
        "Monetary": [5000]
    })

    prediction = predict_new_client(model, new_client)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]




#data frame de test pour la fonction save_model et load_model
def test_model_save_load_predictions(tmp_path):
    X_train = pd.DataFrame({
        "Recency": [10, 20, 100, 150, 30, 200],
        "Frequency": [12, 10, 2, 1, 8, 1],
        "Monetary": [5000, 4000, 300, 200, 3000, 100]
    })

    y_train = [0, 0, 1, 1, 0, 2]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    predictions_before = model.predict(X_train)

    path = tmp_path / "model.pkl" #dossier temporaire pour stocker le modèle
    save_model(model, path)

    loaded_model = load_model(path)
    predictions_after = loaded_model.predict(X_train)

    assert np.array_equal(predictions_before, predictions_after) #verefie que les prédictions avant et après le chargement du modèle sont identiques
