import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation")
st.write("K-Means Clustering using Annual Income & Spending Score")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Feature selection
# -----------------------------
st.subheader("🎯 Select Features for Clustering")

features = st.multiselect(
    "Choose features",
    ["Annual Income (k$)", "Spending Score (1-100)"],
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

if len(features) != 2:
    st.warning("⚠️ Please select exactly 2 features for visualization.")
    st.stop()

X = df[features]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Choose number of clusters
# -----------------------------
k = st.slider("🔢 Number of Clusters (K)", min_value=2, max_value=10, value=5)

# -----------------------------
# Train KMeans
# -----------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📊 Customer Segments")

plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker="X"
)

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("K-Means Customer Segmentation")

st.pyplot(plt)

# -----------------------------
# Cluster details
# -----------------------------
st.subheader("📌 Clustered Data")
st.dataframe(df)

st.success("✅ Clustering Completed Successfully!")
