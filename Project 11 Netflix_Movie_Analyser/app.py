import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="Netflix Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Movie Review Sentiment Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("netflix movie dhurandhar 2.csv", sep=";")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.subheader("Dataset")
st.dataframe(df)

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

with st.spinner("Loading AI model..."):
    classifier = load_model()

if st.button("Analyze Dataset"):
    reviews = df["Review"].astype(str).tolist()

    predictions = classifier(reviews)

    df["Predicted Sentiment"] = [x["label"] for x in predictions]
    df["Confidence"] = [round(x["score"] * 100, 2) for x in predictions]

    st.success("Analysis Completed")
    st.dataframe(df)

st.markdown("---")

st.subheader("Predict Your Own Review")

review = st.text_area("Enter Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = classifier(review)[0]

        if result["label"] == "POSITIVE":
            st.success(f"😊 Positive ({result['score']:.2%})")
        else:
            st.error(f"😞 Negative ({result['score']:.2%})")
