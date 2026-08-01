import streamlit as st
import joblib

model = joblib.load("mental_health_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Mental Health Assistant")

st.title("🧠 Mental Health Assistant")

st.write("Enter your thoughts below. This tool is for educational purposes and does not provide medical diagnosis.")

text = st.text_area("How are you feeling today?")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        st.success(f"Prediction: {prediction}")

        if prediction == "Normal":
            st.info("Maintain your healthy routine.")

        elif prediction == "Stress":
            st.warning("Take regular breaks, rest, and practice relaxation techniques.")

        elif prediction == "Anxiety":
            st.warning("Consider mindfulness exercises or talking with someone you trust.")

        elif prediction == "Depression":
            st.error("This text resembles depression-related language. If these feelings continue, consider speaking with a qualified mental health professional.")

        elif prediction == "Bipolar":
            st.info("This text resembles bipolar-related language. Only a qualified clinician can diagnose mental health conditions.")

        elif prediction == "Personality disorder":
            st.info("This text resembles personality disorder-related language. This model cannot provide a diagnosis.")

        elif prediction == "Suicidal":
            st.error(
                "The model detected language associated with suicidal thoughts. "
                "This prediction is not a diagnosis. If you may be in immediate danger or unable to keep yourself safe, contact your local emergency services or a trusted person right away."
            )