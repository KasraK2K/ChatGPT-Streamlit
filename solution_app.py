# Bring in Deps
import os
from constants import apiKey, organizationId

import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apiKey
os.environ["OPENAI_ORGANIZATION"] = organizationId

# Load text data from a file using TextLoader
local_loader = DirectoryLoader("./data/", glob="*.txt")
local_index = VectorstoreIndexCreator().from_loaders([local_loader])

# Instance
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
wiki = WikipediaAPIWrapper()


def write_response(selected_options):
    if "Local Data" in selected_options:
        local_answer = local_index.query(prompt)
        with st.expander(label="Local Answer:", expanded=True):
            st.write(local_answer)

    if "Local ChatOpenAI" in selected_options:
        chatai_answer = local_index.query(prompt, llm=ChatOpenAI())
        with st.expander(label="Local OpenAI Answer:", expanded=True):
            st.write(chatai_answer)

    if "Local OpenAI" in selected_options:
        openai_answer = local_index.query(prompt, llm=OpenAI())
        with st.expander(label="Local ChatOpenAI Answer:", expanded=True):
            st.write(openai_answer)

    if "Wikipedia" in selected_options:
        wikipedia_answer = wiki.run(prompt)
        with st.expander(label="Wikipedia Answer", expanded=True):
            st.write(wikipedia_answer)

    if "Default" in selected_options:
        chatgpt_answer = llm(prompt)
        with st.expander("OpenAI Answer", expanded=True):
            st.write(chatgpt_answer)


# App Framework
st.markdown(
    body="""<p style="font-size: 46px; font-weight: 900; color: #ff4a5f">
        Embargo <span style="color: #0d243e">AI Model</span></p>
    """,
    unsafe_allow_html=True
)
with st.form("my_form", clear_on_submit=True):
    prompt = st.text_area(label='Ask your question here...', placeholder="Add Text")
    options = st.multiselect(
        label="Please Select Engine:",
        options=["Local Data", "Local ChatOpenAI", "Local OpenAI", "Wikipedia", "Default"],
        default="Local Data"
    )
    submit_button = st.form_submit_button(label="Submit", type="primary", use_container_width=True)

if submit_button and prompt and len(prompt):
    with st.expander(label="Your Question:", expanded=True):
        st.write(prompt)
    write_response(selected_options=options)