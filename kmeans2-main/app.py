import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Iris K-Means", layout="wide")

st.title("🌸 Iris Flower Clustering using K-Means")

# Load Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

st.subheader("Dataset")
st.dataframe(df)

# Select Features
features = st.multiselect(
    "Select Features",
    df.columns,
    default=["petal length (cm)", "petal width (cm)"]
)

if len(features) != 2:
    st.warning("Please select exactly two features.")
    st.stop()

# Scale Data
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])

# Number of clusters
k = st.slider("Number of Clusters", 2, 6, 3)

# Train Model
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X)

# Show clustered data
st.subheader("Clustered Data")
st.dataframe(df)

# Plot
fig, ax = plt.subplots(figsize=(7,5))

for cluster in range(k):
    temp = df[df["Cluster"] == cluster]
    ax.scatter(
        temp[features[0]],
        temp[features[1]],
        label=f"Cluster {cluster}"
    )

centers = scaler.inverse_transform(model.cluster_centers_)

ax.scatter(
    centers[:,0],
    centers[:,1],
    marker="X",
    s=200,
    color="black",
    label="Centroids"
)

ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.legend()

st.pyplot(fig)

# Elbow Method
st.subheader("Elbow Method")

sse = []

for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    sse.append(km.inertia_)

fig2, ax2 = plt.subplots()

ax2.plot(range(1,11), sse, marker="o")
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("SSE")
ax2.set_title("Elbow Curve")

st.pyplot(fig2)
