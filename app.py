import os
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'RAGu eMLLiano'

@app.route('/chat', methods=['GET'])
def chat():
	q = request.args.get('q')
    return q

if __name__ == '__main__':
    # Port number is required to fetch from env variable
    # http://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#PORT
    CF_PORT = os.getenv("PORT", default=8000)
    app.run(host='0.0.0.0', port=int(CF_PORT), debug=True)