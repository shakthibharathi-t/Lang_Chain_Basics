import os, pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Automatically create dataset and files
os.makedirs("dataset", exist_ok=True)
pd.DataFrame([{"id": 1, "query": "What are the library timings?"}]).to_csv("dataset/user_queries.csv", index=False)
pd.DataFrame([{"id": 1, "prompt": "Answer concisely: {query}"}]).to_csv("dataset/prompts.csv", index=False)

# 2. Run LangChain with Ollama
llm = OllamaLLM(model="llama3.2")
chain = PromptTemplate.from_template("Answer concisely: {query}") | llm | StrOutputParser()

# 3. Process query and save results
res = chain.invoke({"query": "What are the library timings?"})
pd.DataFrame([{"id": 1, "response": res}]).to_csv("dataset/chatbot_responses.csv", index=False)
pd.DataFrame([{"query": "What are the library timings?", "response": res}]).to_csv("dataset/chain_results.csv", index=False)

print("Done! All files created and saved successfully.")