from flask import Flask,jsonify,request,Response
import json
from main import check
import numpy as np

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/recognize', methods=['GET'])
def recognize():
    dict = request.args.to_dict()
    arr = np.array(json.loads(dict['data']))
    print(arr.shape)
    digit = check(arr)

    resp = Response(str(digit))
    # resp.headers['Access-Control-Allow-Origin'] = '*'

    return resp

if __name__ == '__main__':
    app.run()