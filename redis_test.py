from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
import os

os.environ["OPENAI_API_KEY"] = ""
REDIS_URL = "redis://localhost:6379"

model = ChatOpenAI(model_name="gpt-3.5-turbo")

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a trustworthy AI assistant. Answer the question below. \
                    If your don't know, just say you don't know it."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

chain = template | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "foo"}}

# chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

print(chain_with_history.invoke({"question": "Do you remember me who i am?"}, config=config))

# history = RedisChatMessageHistory("foo", url=REDIS_URL)
# history.add_user_message("hi!")
# history.add_ai_message("what's up?")
#
# print(history.messages)
#
# model = ChatOpenAI(model_name="gpt-3.5-turbo")


