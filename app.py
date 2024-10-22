import os
from pathlib import Path

from decouple import config
from flask import Flask, request, render_template, flash, redirect
from flask.json import jsonify
from flask_cors import CORS


app = Flask(__name__)
app.secret_key = config('SECRET_KEY', default='super secret key')
app.config['PROPAGATE_EXCEPTIONS'] = config('DEBUG', cast=bool, default=True)
app.config['UPLOAD_FOLDER'] = config('UPLOAD_FOLDER', default='upload/')
app.config['SESSION_TYPE'] = 'filesystem'
CORS(app, resources={r"/api/*": {"origins": "*"}})

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

        db = HanaDB(embedding=EMBEDDINGS, connection=connection,
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

def _chat(question):
    from langchain.chains import RetrievalQA
    init_vector_db()
    retriever = DB.as_retriever(search_kwargs={'k':20})
    qa = RetrievalQA.from_chain_type(llm=get_llm(),
                     retriever=retriever,
                     chain_type='stuff',
                     chain_type_kwargs={'prompt': get_prompt()})
    answer = qa.invoke(question)
    return answer

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
        Exemple:
        url -X POST http://127.0.0.1:5000/api/chat -d '{"question": "What are the healthcare benefits?"}'
    """
    data = request.get_json(force=True)
    question = data.get('question', '') or data.get('q', '')
    if not question:
        return jsonify({'msg': 'Please, use the JSON {"question": ""} format to interact.'}), 400
    return _chat(question)

@app.route('/chat', methods=['GET', 'POST'])
def ui_chat():
    question, answer = '', ''
    if request.method == 'POST':
        question = request.form['question']
        ivk = _chat(question)
        question, answer = ivk['query'], ivk['result']

    return render_template('chat.html',
        question=question,
        answer=answer)

def get_uploaded_files():
    p = Path(app.config['UPLOAD_FOLDER']).glob('**/*')
    return [x for x in p if x.is_file() and not x.name.startswith('.')]

def _upload(request):
    """
        From: https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads/
    """
    from werkzeug.utils import secure_filename

    print(request.files)

    # check if the post request has the file part
    if 'file' not in request.files:
        raise Exception('No file part')

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    file = request.files['file']
    if file.filename == '':
        raise Exception('No selected file')

    suffix = Path(file.filename).suffix
    if not suffix in ['.pdf']:
        raise Exception(f'Filetype is not supported: {suffix}')

    filename = secure_filename(file.filename)
    file.save(Path(app.config['UPLOAD_FOLDER']) / filename)

    return filename

@app.route('/api/files', methods=['GET', 'POST', 'DELETE'])
def api_upload():
    """
        Exemple:
        curl http://127.0.0.1:5000/api/files
        curl -X POST -H "Content-Type: multipart/form-data" -F "file=@myfile.pdf" http://127.0.0.1:5000/api/files
        curl -X DELETE http://127.0.0.1:5000/api/files
    """

    http_code, msg = 200, ''
    if request.method == 'POST':
        try:
            _upload(request)
            http_code = 201
        except Exception as e:
            http_code, msg = 400, str(e)
 
    files = get_uploaded_files()

    if request.method == 'DELETE':
        for f in files:
            f.unlink(missing_ok=True)
        http_code, msg = 205, 'Files Reseted!'

    return jsonify({'msg': msg, 'files': [str(f) for f in files]}), http_code

@app.route('/upload', methods=['GET', 'POST'])
def ui_upload():

    if request.method == 'POST':
        try:
            _upload(request)
            flash(f'File Upload! ✌️', 'success')
        except Exception as e:
            flash(str(e), 'error')
            return redirect(request.url)

    return render_template('upload.html',
            uploaded_files=get_uploaded_files())

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        init_vector_db()
        DB.delete(filter={})
        for f in get_uploaded_files():
            f.unlink(missing_ok=True)
        flash(f'HANA and Files Reseted! ✌️', 'success')

    return render_template('reset.html',
            uploaded_files=get_uploaded_files())

def _embed(uploaded_files = get_uploaded_files()):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    init_vector_db()

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

@app.route('/api/embed_docs', methods=['PUT', 'DELETE'])
def api_embed_docs():

    if request.method == 'DELETE':
        init_vector_db()
        DB.delete(filter={})
        http_code, msg = 205, 'HANA DB Reseted'
   
    if request.method == 'PUT':
        _embed()
        http_code, msg = 202, 'Documents embedded into HANA DB.'

    return jsonify({'msg': msg}), http_code

@app.route('/embed_docs', methods=['GET', 'POST'])
def ui_embed_docs():

    uploaded_files = get_uploaded_files()
    if request.method == 'POST':
        _embed(uploaded_files)
        flash(f'Files embeded into HANA! ✌️', 'success')

    return render_template('embed_docs.html',
            uploaded_files=uploaded_files)

if __name__ == '__main__':
    # Port number is required to fetch from env variable
    # http://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#PORT
    cf_port = config('PORT', cast=int, default=8000)
    app.run(host='0.0.0.0', port=cf_port, debug=True)
