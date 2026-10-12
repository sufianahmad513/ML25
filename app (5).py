import streamlit as st
import pickle

# Set page title and styling
st.set_page_config(page_title="Email Fraud & Spam Detector", page_icon="✉️", layout="centered")

st.title("✉️ Email Fraud & Spam Detector")
st.write("This application uses a Naive Bayes Classifier trained on email texts to detect potential fraud, phishing, or spam.")

# Load the saved model and vectorizer
@st.cache_resource
def load_artifacts():
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    st.success("Model artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")

# Input area
user_input = st.text_area("Paste the email content below:", height=150, placeholder="Type or paste your email here...")

if st.button("Analyze Email", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some email content first.")
    else:
        # Vectorize and predict
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        confidence = model.predict_proba(text_vec)[0]
        prob = confidence[prediction] * 100

        # Display result
        st.subheader("Analysis Result:")
        if prediction == 1:
            st.error(f"🔴 FRAUD DETECTED ({prob:.2f}% Confidence)")
            st.write("**Recommendation:** Do not click on any links, download attachments, or provide personal details.")
        else:
            st.success(f"🟢 CLEAN EMAIL ({prob:.2f}% Confidence)")
            st.write("**Recommendation:** This email looks safe, but always practice caution with unfamiliar senders.")
