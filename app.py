import streamlit as st
import boto3

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Bedrock
from langchain_experimental.chat_models import Llama2Chat

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

# Set default AWS region
boto3.setup_default_session(region_name='us-east-1')

st.set_page_config(page_title="JaredsChatbotTutorial", page_icon="🦒")
st.title('Jared ❤️s ML')

session_id = '1'


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def create_chain(model="meta.llama2-13b-chat-v1",
                 temperature=0.01,
                 max_gen_len=1000,
                 system_prompt="You are a smart assistant.",
                 callbacks=None):
    llm = Bedrock(
        model_id="meta.llama2-13b-chat-v1",
        model_kwargs={"temperature": temperature,
                      "max_gen_len": max_gen_len},
        streaming=True,
    )

    template_messages = [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    model = Llama2Chat(llm=llm, callbacks=callbacks)
    chain = prompt_template | model
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
    )


# Initialize DynamoDBChatMessageHistory
# boto3_session = Session(region_name='us-east-1')
msgs = DynamoDBChatMessageHistory(
    table_name="ChatbotSessionTable", 
    session_id=session_id
)


with st.sidebar:
    model = st.selectbox(
        'Model',
        ['meta.llama2-13b-chat-v1', 'meta.llama2-70b-chat-v1'],
        index=0  # Default to the first model
    )
    temperature = st.slider('Temperature', 0.0, 1.0, 0.01)  # Default temperature is 0.01
    max_gen_len = st.slider('Max Generation Length', 1, 2048, 1000)  # Default max_gen_len is 1000
    system_prompt = st.text_input('System Prompt', 'You are a smart assistant.')  # Default system prompt
    if st.button('Clear'):
        msgs.clear()  # Clear messages when button is pressed


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        chain = create_chain(model, temperature, max_gen_len, system_prompt, callbacks=[stream_handler])
        config = {"configurable": {"session_id": session_id}}
        response = chain.invoke({"input": prompt}, config)
        
