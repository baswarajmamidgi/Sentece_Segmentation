import sys
import traceback
import json

import numpy as np
import statistics
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

model_file_name = 'disease_classifier.pkl'

clf = joblib.load(model_file_name)
query=[]

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            query=[]
            json_ = request.json
            #print (json_)
            #query=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1]]
            #query = pd.DataFrame(json_)


            data=[]
            #print dict.values(json_[0])
            for key ,val in dict.items(json_[0]):
                #print('a   '+val)

                val=[[float(s) for s in val.split(',')]]
                data.append(val)


            query=np.vstack(data)
            print (np.shape(query))

            # https://github.com/amirziai/sklearnflask/issues/3

            prediction = list(clf.predict(query))
            try:
                prediction= statistics.mode(prediction)
            except:
                prediction=-1

            return jsonify({'prediction': prediction})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print ('train first')
        return ('no model here')





if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 8080

    try:
        clf = joblib.load(model_file_name)
        print ('model loaded')
        #model_columns = joblib.load(model_columns_file_name)
        print ('model columns loaded')

    except Exception as e:
        print ('No model here')
        print ('Train first')
        print (str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)