#https://rfriend.tistory.com/838
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun

import os
os.environ["OPENAI_API_KEY"] = ""

search = DuckDuckGoSearchRun()

template = """You are an AI assistant. Answer the question.
If you don't know the answer, just say you don't know.

Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model_name="gpt-3.5-turbo")
parser = StrOutputParser()

chain = (
    {"question": RunnablePassthrough(), "context": RunnablePassthrough}
    | prompt
    | model
    | parser
)

# chain.invoke({"question": "Which BTS member was the last to go to the military?"})
question = "Which BTS member was the last to go to the military?"

print(chain.invoke({"question": question, "context": search.run(question)}))



