import re 
import streamlit as st
import requests 


# WEBHOOK_URL = st.secrets["WEBHOOK_URL"]
WEBHOOK_URL = "https://connect.pabbly.com/workflow/sendwebhookdata/IjU3NjUwNTY0MDYzMDA0MzM1MjY4NTUzNzUxMzci_pc"

def is_valid_email(email):
    # Basic regex pattern for email validation
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_pattern, email) is not None

def contact_form():
    with st.form("contact_form"):
        name = st.text_input("First name")
        email = st.text_input("Email Adress")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if not WEBHOOK_URL:
                st.error("Email service is not set up. Please try again later.",icon="📧")
                st.stop()

            if not name:
                st.error("Please provide your name.", icon="👤")
                st.stop()

            if not email:
                st.error("Please provide your email.", icon="📧")
                st.stop()

            if not is_valid_email(email):
                st.error("Please provide a valid email address.", icon="📧")
                st.stop()

            if not message:
                st.error("Please provide your message.", icon="💬")
                st.stop()

            # Prepare the data payload and send it to the specified webhook URL
            data = {"email": email, "name": name, "mmessage": message}
            response = requests.post(WEBHOOK_URL, json=data)

            if response.status_code == 200:
                st.success("Your message has been sent successfully!", icon="🎉")
            else:
                st.error("There was an error sending your message.",icon="🙁")





