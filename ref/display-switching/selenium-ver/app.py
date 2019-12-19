from flask import Flask
from flask import send_file

app = Flask(__name__)

@app.route('/')
def get_sample():
    return send_file('sample.gif', mimetype='image/gif')

@app.route('/sample2')
def get_sample2():
    return send_file('sample2.gif', mimetype='image/gif')

@app.route('/sample3')
def get_sample3():
    return send_file('sample3.gif', mimetype='image/gif')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)