# Bring in Deps
import os
from constants import apiKey, organizationId

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apiKey
os.environ["OPENAI_ORGANIZATION"] = organizationId

# # App Framework
st.markdown('<p style="font-size: 32px; font-weight: 600; color: #ff4a5f">Embargo AI Model</p>', unsafe_allow_html=True)
prompt = st.text_input('Ask your question here...')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic', 'wikipedia_research'],
    template="""
        write me a youtube video title about {topic}
        while leveraging this wikipedia research: {wikipedia_research}
     """
)

script_template = PromptTemplate(
    input_variables=['title'],
    template='write me a youtube video script base on this title TITLE: {title}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Instance
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    title = title_chain.run(topic=prompt, wikipedia_research=prompt)
    wikipedia_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wikipedia_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wikipedia_research)