import os

from flask import Flask, request
app = Flask(__name__)

# Generative AI Config
from config_creator import create_aicore_config
create_aicore_config()

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
LLM = ChatOpenAI(proxy_model_name='gpt-35-turbo', proxy_client=proxy_client)

@app.route('/')
def hello():
    return 'RAGu eMLLiano'

def embeddings():
    from hdbcli import dbapi
    from langchain_community.vectorstores.hanavector import HanaDB
    from gen_ai_hub.proxy.langchain.init_models import init_embedding_model

    embeddings = init_embedding_model('text-embedding-ada-002')
    connection = dbapi.connect(
        os.getenv("SAP_HANA_HOST"),
        port="443",
        user=os.getenv("SAP_HANA_USER"),
        password=os.getenv("SAP_HANA_PASS"),
        autocommit=True,
        sslValidateCertificate=False,
    )
    db = HanaDB(embedding=embeddings, connection=connection,
        table_name="CATALOG_UPDATED_DEV_1_"+ hdb_user)

    return db

def get_prompt():
    from langchain.prompts.chat import
    # TODO
    prompt_template = "You are a helpful AI bot. "
        "Context: {context} Question: {question}"
    return PromptTemplate(
        template = prompt_template, 
        input_variables=["context", "question"])

@app.route('/chat', methods=['GET'])
def chat():
    question = request.args.get('q')

    db = embeddings()
    retriever = db.as_retriever(search_kwargs={'k':20})

    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=LLM,
                     retriever=retriever, 
                     chain_type="stuff",
                     chain_type_kwargs={"prompt": get_prompt()})

    answer = qa.run(question)
    return answer

if __name__ == '__main__':
    # Port number is required to fetch from env variable
    # http://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#PORT
    cf_port = os.getenv("PORT", default=8000)
    app.run(host='0.0.0.0', port=int(cf_port), debug=True)