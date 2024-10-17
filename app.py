import os
from pathlib import Path

from decouple import config
from flask import Flask, request, render_template, flash

app = Flask(__name__)
app.secret_key = config('SECRET_KEY', default='super secret key')
app.config['PROPAGATE_EXCEPTIONS'] = config('DEBUG', cast=bool, default=True)
app.config['UPLOAD_FOLDER'] = config('UPLOAD_FOLDER', default='upload/')
app.config['SESSION_TYPE'] = 'filesystem'

DB, EMBEDDINGS = None, None

def init_vector_db():
    global DB, EMBEDDINGS
    if EMBEDDINGS is None:
        from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
        #embeddings = init_embedding_model('text-embedding-ada-002')
        embeddings = init_embedding_model('text-embedding-3-small')
        # DOCS: check the ~/.aicore/config.json
        EMBEDDINGS = embeddings

    if DB is None:
        from hdbcli import dbapi
        from langchain_community.vectorstores.hanavector import HanaDB
        hdb_user = config('SAP_HANA_USER')
        connection = dbapi.connect(
            config('SAP_HANA_HOST'),
            port=443,
            user=hdb_user,
            password=config('SAP_HANA_PASS'),
            autocommit=True,
            #sslValidateCertificate=False,
        )

        db = HanaDB(embedding=embeddings, connection=connection,
            table_name='CATALOG_UPDATED_DEV_1_' + hdb_user)
        DB = db

        return DB, EMBEDDINGS

def get_llm():
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

    proxy_client = get_proxy_client('gen-ai-hub')
    #return ChatOpenAI(proxy_model_name='gpt-35-turbo', proxy_client=proxy_client)
    return ChatOpenAI(proxy_model_name='gpt-4o-mini', proxy_client=proxy_client)

def get_prompt():
    from langchain.prompts import PromptTemplate
    prompt_template = 'You are a helpful AI bot. Context: {context} Question: {question}'
    return PromptTemplate(
        template = prompt_template,
        input_variables=['context', 'question'])

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/chat', methods=['GET'])
def chat():
    question = request.args.get('q')
    if not question:
        return 'Please, use the ?q= to send questions.'

    init_vector_db()
    retriever = DB.as_retriever(search_kwargs={'k':20})

    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=get_llm(),
                     retriever=retriever,
                     chain_type='stuff',
                     chain_type_kwargs={'prompt': get_prompt()})

    answer = qa.invoke(question)
    return answer

def get_uploaded_files():
    p = Path(app.config['UPLOAD_FOLDER']).glob('**/*')
    return [x for x in p if x.is_file() and not x.name.startswith('.')]

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
        From: https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads/
    """
    from werkzeug.utils import secure_filename
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        suffix = Path(file.filename).suffix
        if not suffix in ['.pdf']:
            flash(f'Filetype is not supported: {suffix}', 'error')

        filename = secure_filename(file.filename)
        file.save(Path(app.config['UPLOAD_FOLDER']) / filename)
        flash(f'File Upload! ✌️', 'success')

    return render_template('upload.html',
            uploaded_files=get_uploaded_files())

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        init_vector_db()
        DB.delete(filter={})
        for f in get_uploaded_files():
            f.unlink()
        flash(f'HANA and Files Reseted! ✌️', 'success')

    return render_template('reset.html',
            uploaded_files=get_uploaded_files())


@app.route('/embed_docs', methods=['GET', 'POST'])
def embed_docs():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    uploaded_files=get_uploaded_files()

    if request.method == 'POST':

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        init_vector_db()
        DB.delete(filter={})

        for filepath in uploaded_files:
            documents = []
            loader = PyPDFLoader(filepath)
            for n, doc_page in enumerate(loader.lazy_load()):
                print(f'Embeddings {filepath}, page {n}')
                documents.append(doc_page)

            print(f'Uploading documents to HANA DB...')
            text_chunks = text_splitter.split_documents(documents)
            DB.add_documents(text_chunks)
            print(f'Uploading DONE!')

        flash(f'Files embeded into HANA! ✌️', 'success')

    return render_template('embed_docs.html',
            uploaded_files=uploaded_files)

if __name__ == '__main__':
    # Port number is required to fetch from env variable
    # http://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#PORT
    cf_port = config('PORT', cast=int, default=8000)
    app.run(host='0.0.0.0', port=cf_port, debug=True)
