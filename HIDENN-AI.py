import streamlit as st
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Welcome to HIDENN-AI, LLC",
    page_icon="🌐",
    layout="centered",
)

# Display logo in the upper left corner
logo = Image.open("logo.jpg")
col1, col2 = st.columns([1, 8]) 
with col1:
    st.image(logo, use_container_width=True)
with col2:
    st.title("Welcome to HIDENN-AI, LLC") 

# Add introductory description
st.markdown(
    """
    ### Who We Are
    HIDENN-AI, LLC is a cutting-edge technology startup co-founded by researchers at Northwestern University (led by Dr. Wing Kam Liu)
    and UT Dallas (led by Dr. Dong Qian).

    ### Our Vision
    At HIDENN-AI, We are developing a next generation software that combines physics with AI 
    to create and analyze complex digital twins in design and manufacturing.


    ### Our Services
    - Faster and more accurate Finite Element Analysis (C-HiDeNN Module)
    - Resource-efficient AI surrogate model training (INN trainer)
    - Data-free surrogate modeling (Space-Time-Parameter C-HiDeNN-TD solver)

    Join us on our journey to transform the future with AI!

    ### Visit Our Homepage
    [HIDENN-AI Homepage](https://hidennai.wordpress.com/)
    """
)

# Add a footer
st.write("---")
st.caption("© 2024 HIDENN-AI, LLC. All Rights Reserved.")