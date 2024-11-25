import streamlit as st
st.title("Mon premier Streamlit")
st.header("header")
st.subheader("title") 
st.write("Introduction")
if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")
  st.code()