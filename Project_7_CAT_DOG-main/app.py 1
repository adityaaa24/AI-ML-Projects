import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("K-Means Clustering Demo")

# Load Dataset
df = pd.read_csv("income.csv")

st.subheader("Dataset")
st.dataframe(df)

# Select Features
features = st.multiselect(
    "Select Features",
    df.columns,
    default=["Age", "Income($)"]
)

if len(features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

X = df[features]

# Scale Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Number of Clusters
k = st.slider("Number of Clusters", 2, 10, 3)

# Train Model
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)

st.subheader("Clustered Data")
st.dataframe(df)

# Plot
fig, ax = plt.subplots(figsize=(8,6))

for cluster in range(k):
    cluster_data = df[df["Cluster"] == cluster]
    ax.scatter(
        cluster_data[features[0]],
        cluster_data[features[1]],
        label=f"Cluster {cluster}"
    )

centers = scaler.inverse_transform(model.cluster_centers_)
ax.scatter(
    centers[:,0],
    centers[:,1],
    color="black",
    marker="X",
    s=200,
    label="Centroids"
)

ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.legend()

st.pyplot(fig)
