Python 3.10.0 (tags/v3.10.0:b494f59, Oct  4 2021, 19:00:18) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', " ", text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words("english")]
    text = ' '.join(cleaned)
    return text

st.title("Identifying Review's Rating")
st.header("Instructions")
st.markdown("1.Review column's name should be **Text**")
st.markdown("2.Rating column's name should be **Star**")
st.markdown("3.Rating range should be 0-5")

uploaded_file = st.file_uploader("Choose a File")

data = pd.read_csv(uploaded_file)
st.write(data)

if st.button("Click for Results") :
    data["Text"] = data["Text"].apply(lambda x: clean_text(str(x)))

    sid = SentimentIntensityAnalyzer()

    data["sentiment_Score"] = data["Text"].apply(lambda review:sid.polarity_scores(review))
    data["sentiment_Compound_Score"]  = data['sentiment_Score'].apply(lambda x: x['compound'])
    data["Review_type"] = data["sentiment_Compound_Score"].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
    st.bar_chart(data.Review_type.value_counts())

    positive_review = data[(data.Review_type == "positive")]
    positive_review["Opinion"] = positive_review["Star"].apply(lambda star: "No Attention Needed" if star >= 3 else "Attention Needed")
    st.bar_chart(positive_review.Opinion.value_counts())

    data_frame = positive_review

    st.download_button(
        label="Download data as CSV",
        data_frame=data.to_csv().encode("utf-8"),
        file_name='data.csv',
        mime='text/csv',
    )
