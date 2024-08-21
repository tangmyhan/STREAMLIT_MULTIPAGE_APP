import streamlit as st

from forms.contact import contact_form

@st.dialog("Contact Me")  # popup
def show_contact_form():
    contact_form()

col1, col2 = st.columns(2, gap="small",vertical_alignment="center")
with col1:
    st.image("./assets/anhfb.jpg", width=230)
with col2:
    st.title("Tang My Han", anchor=False)
    st.write(
        """
        Hi everyone, My name is Tang My Han from Ho Chi Minh City, Viet Nam.
        I am a student in Computer Science at University of Information Technology.
        """
    )
    if st.button("📩 Contact Me"):
        show_contact_form()

# --- 
st.write("\n")
# st.subheader("Experience & Qualifications", anchor=False)
# st.write(
#     """
#     ...
#     """
# )
