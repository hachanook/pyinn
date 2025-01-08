import streamlit as st
import pandas as pd
import os


# Initialize session state for button states
if 'button1_pressed' not in st.session_state:
    st.session_state.button1_pressed = False
if 'button2_pressed' not in st.session_state:
    st.session_state.button2_pressed = False
if 'button3_pressed' not in st.session_state:
    st.session_state.button3_pressed = False


st.button("Reset", type="primary")
if st.button("Say hello") or st.session_state.button1_pressed:
    st.session_state.button1_pressed = True
    st.write("Why hello there")
else:
    st.write("Goodbye")

if st.button("Aloha", type="tertiary"):
    st.session_state.button2_pressed = True
    st.write("Ciao")