import streamlit as st

# --- PAGE SETUP ---
about_page = st.Page(
    page="views/about_me.py",
    title="About Me",
    icon=":material/account_circle:",
    default=True, 
)

project_1_page = st.Page(
    page="views/Iris.py",
    title="Iris Flower Prediction",
    icon=":material/bar_chart:",
)

project_2_page = st.Page(
    page="views/bg_remove.py",
    title="Image Background Remover",
    icon=":material/replace_image:"

)

project_3_page = st.Page(
    page="views/hyperparameter_tuning.py",
    title="Hyperparameter Tuning",
    icon=":material/tune:",
)

project_4_page = st.Page(
    page="views/chatbot.py",
    title="Chat Bot",
    icon=":material/smart_toy:",
)





# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])


# --- NAVIGATION SETUP [WITH SECTIONS] ---
pg = st.navigation(
    {
        "Info": [about_page],
        "Project": [project_1_page, project_2_page, project_3_page, project_4_page]
    }
)

# --- RUN NAVIGATION ---
pg.run()

