import streamlit as st
import joblib 
import re 
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_artifacts():
    cat_model = joblib.load('category_model.pkl')
    prio_model = joblib.load('priority_model.pkl')
    tf_idf = joblib.load('tfidf_vectorizer.pkl')
    X_train_tf_idf = joblib.load('X_train_tf_idf.pkl')
    X_train = joblib.load('X_train.pkl')
    y_train = joblib.load('y_train.pkl')
    return cat_model, prio_model, tf_idf, X_train_tf_idf, X_train, y_train

cat_model, prio_model, tf_idf, X_train_tf_idf, X_train, y_train = load_artifacts()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

st.set_page_config(page_title='Ticket Classification System', page_icon='*')
st.title("Customer support ticket classifier")
st.write("Enter the text of ticket to classify the problem, predict priority, define similarity tickets")

user_input = st.text_area("Customer Message:", "I was charged twice on my credit card for the same order, please refund me.")

if st.button("Analyze Ticket"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        
        vec_input = tf_idf.transform([cleaned])

        pred_category = cat_model.predict(vec_input)[0]
        pred_priority = prio_model.predict(vec_input)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Predicted Category:** {pred_category}")
        with col2:
            st.warning(f"**Predicted Priority:** {pred_priority}")
        
        similarities = cosine_similarity(vec_input, X_train_tf_idf).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        
        st.subheader("Top 3 Similar Past Tickets:")
        for i, idx in enumerate(top_indices, 1):
            with st.expander(f"{i}. Category: {y_train.iloc[idx]} (Similarity Score: {similarities[idx]:.4f})"):
                st.write(X_train.iloc[idx])
    else:
        st.error("الرجاء كتابة نص التذكرة أولاً.")