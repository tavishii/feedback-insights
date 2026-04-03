import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

st.title("Smart Feedback Sentiment Analyzer")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    # Load dataset
    df = pd.read_csv(file, header=None)
    df.columns = ['rating', 'title', 'text']

    # Clean data
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Convert rating → sentiment (1=Negative, 2=Positive)
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x == 2 else 0)

    X = df['text']
    y = df['sentiment']

    # Vectorize text
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Accuracy")
    st.write(accuracy)

    # Predict full dataset
    df['Predicted Sentiment'] = model.predict(X_vectorized)
    df['Predicted Sentiment'] = df['Predicted Sentiment'].map({1: "Positive", 0: "Negative"})

    st.subheader("Analyzed Data")
    st.write(df)

    # Pie chart
    counts = df['Predicted Sentiment'].value_counts()

    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)

    # User input review
    st.subheader("Try Your Own Review")
    user_input = st.text_area("Enter your review:")
    if st.button("Analyze"):
        if user_input:
            input_vec = vectorizer.transform([user_input])
            result = model.predict(input_vec)[0]
            if result == 1:
                st.success("Positive Review 😊")
            else:
                st.error("Negative Review 😡")

    # Download report
    st.subheader("Download Report")
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    total = pos + neg
    pos_percent = (pos / total) * 100 if total > 0 else 0
    neg_percent = (neg / total) * 100 if total > 0 else 0

    report = f"""
SMART SENTIMENT ANALYSIS REPORT

Total Reviews: {total}

Positive Reviews: {pos} ({pos_percent:.2f}%)
Negative Reviews: {neg} ({neg_percent:.2f}%)

Key Insights:
- If negative reviews are high, improvement is needed.
- Focus on customer complaints and feedback.

Conclusion:
The system analyzed customer sentiment and provided insights for decision-making.
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="sentiment_report.txt",
        mime="text/plain"
    )