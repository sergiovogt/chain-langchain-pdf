import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from backend import chain

st.title("💬 Chat con contenido de un PDF ")
st.subheader("🍕 Recetas de cocina y nutrición (Cocina y nutrición.pdf)")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = chain.invoke(
            prompt
        )
        st.write(response)









