import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
sns.set_style("whitegrid")

st.set_page_config(page_title="Product Review Sentiment Dashboard", layout="wide")
st.title("🚀 Product Review Sentiment Analysis Dashboard")
st.markdown("Analyzing product reviews from `Reviews.csv` hosted on GitHub!")

url = "https://raw.githubusercontent.com/nithinarchive/Product-Review-Classifier-using-Sentiment-Analysis/main/Reviews.csv"
try:
    df = pd.read_csv(url)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Auto-detect text column for reviews
text_col = None
for col in df.columns:
    if df[col].dtype == object:
        text_col = col
        break

if not text_col:
    st.error("No text/review column found in CSV!")
    st.stop()

df['Review'] = df[text_col].astype(str)
df['Sentiment'] = df['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Ensure rating column is numeric and clean
rating_col = None
if 'rating' in df.columns or 'Rating' in df.columns:
    rating_col = 'rating' if 'rating' in df.columns else 'Rating'
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    df = df.dropna(subset=[rating_col, 'Sentiment'])
else:
    df = df.dropna(subset=['Sentiment'])

st.subheader("Preview of Data")
st.dataframe(df.head())

# Sidebar filters
st.sidebar.header("Filters")
sentiments = st.sidebar.multiselect(
    "Select Sentiments:", options=df['Sentiment_Label'].unique(), default=df['Sentiment_Label'].unique()
)

if rating_col:
    rating_min, rating_max = st.sidebar.slider(
        "Select Rating Range:",
        float(df[rating_col].min()),
        float(df[rating_col].max()),
        (float(df[rating_col].min()), float(df[rating_col].max()))
    )
    df_filtered = df[(df['Sentiment_Label'].isin(sentiments)) &
                     (df[rating_col] >= rating_min) & (df[rating_col] <= rating_max)]
else:
    df_filtered = df[df['Sentiment_Label'].isin(sentiments)]

st.subheader("Filtered Data")
st.dataframe(df_filtered)

# Metrics
st.subheader("📊 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(df_filtered))
col2.metric("Positive Reviews", len(df_filtered[df_filtered['Sentiment_Label'] == 'Positive']))
col3.metric("Negative Reviews", len(df_filtered[df_filtered['Sentiment_Label'] == 'Negative']))
col4.metric("Neutral Reviews", len(df_filtered[df_filtered['Sentiment_Label'] == 'Neutral']))

# Sentiment Distribution
st.subheader("📈 Sentiment Distribution")
sentiment_counts = df_filtered['Sentiment_Label'].value_counts()
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', ax=ax1)
ax1.set_ylabel("Count")
ax1.set_xlabel("Sentiment")
st.pyplot(fig1)

# Word Cloud
st.subheader("☁️ Word Cloud")
all_text = ' '.join(df_filtered['Review'].tolist())
if all_text.strip() != "":
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)
else:
    st.info("No reviews to generate Word Cloud.")

# Rating vs Sentiment
if rating_col:
    st.subheader("📊 Rating vs Sentiment")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.boxplot(x=rating_col, y='Sentiment', hue='Sentiment_Label', data=df_filtered, palette='viridis', ax=ax3)
    ax3.set_title("Rating vs Sentiment Distribution")
    st.pyplot(fig3)

# Top Reviews
st.subheader("🌟 Top Positive Reviews")
st.write(df_filtered.sort_values(by='Sentiment', ascending=False)[['Review', 'Sentiment']].head(5))
st.subheader("💀 Top Negative Reviews")
st.write(df_filtered.sort_values(by='Sentiment', ascending=True)[['Review', 'Sentiment']].head(5))
