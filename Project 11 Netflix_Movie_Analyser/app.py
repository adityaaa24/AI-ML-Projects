import streamlit as st
import pandas as pd
from transformers import pipeline
import evaluate

st.set_page_config(
    page_title="Netflix Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Movie Review Sentiment Analysis")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("netflix movie dhurandhar 2.csv", delimiter=";")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df)

reviews = df["Review"].tolist()
labels = df["Class"].tolist()

# Load Model
@st.cache_resource
def load_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier

classifier = load_model()

# Predict Dataset
if st.button("Analyze Dataset"):

    predictions = classifier(reviews)

    result_df = pd.DataFrame({
        "Review": reviews,
        "Actual": labels,
        "Predicted": [p["label"] for p in predictions],
        "Confidence": [round(p["score"],4) for p in predictions]
    })

    st.success("Analysis Completed")
    st.dataframe(result_df)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    refs = [1 if x=="POSITIVE" else 0 for x in labels]
    preds = [1 if x["label"]=="POSITIVE" else 0 for x in predictions]

    acc = accuracy.compute(references=refs,predictions=preds)["accuracy"]
    f1_score = f1.compute(references=refs,predictions=preds)["f1"]

    c1,c2 = st.columns(2)

    c1.metric("Accuracy",f"{acc:.2%}")
    c2.metric("F1 Score",f"{f1_score:.2%}")

# Single Prediction
st.divider()

st.subheader("Try Your Own Review")

user_review = st.text_area("Write a movie review")

if st.button("Predict Sentiment"):

    if user_review.strip():

        pred = classifier(user_review)[0]

        st.write("### Prediction")

        if pred["label"]=="POSITIVE":
            st.success(f"😊 {pred['label']}")
        else:
            st.error(f"☹️ {pred['label']}")

        st.write(f"Confidence : {pred['score']:.2%}")

    else:
        st.warning("Please enter a review.")
