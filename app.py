import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle



from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback

application = Flask(__name__)


#загружаем модели из файла
vec = pickle.load(open("./models/tfidf.pickle", "rb"))
model = lgb.Booster(model_file='./models/lgbm_model_v2.txt')


# тестовый вывод
@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    
    response = jsonify(resp)
    
    return response

# предикт категории
@application.route("/categoryPrediction", methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'}
    try:
        #Получение данных
        getData = request.get_data()
        getData = getData.decode("utf-8")
        #Разбиение на разные запросы
        arr=getData.split('\r')
        i=0
        for tmp in arr:
            if tmp =='':
                continue
            tmp = tmp.replace("'","\"")
            json_params = json.loads(tmp)
            #Предикт категории сообщений (выдает вероятность на принадлежность к классу)
            category = model.predict(vec.transform([json_params['user_message']]).toarray()).tolist()
            resp['category_message_'+str(i)] = category
            i+=1       
    except Exception as e: 
        print(e)
        resp['message'] = e
    
    response = jsonify(resp)
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)