import flask
from flask import request
# import model

# 
# from pythainlp.tokenize import word_tokenize
# from gensim.models import KeyedVectors
# import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
from statistics import mode 
# import tensorflow as tf

# import dill as pickle
import pandas as pd
# from pythainlp import word_vector
from pythainlp.word_vector import sentence_vectorizer

# from torch import  from_numpy,cosine_similarity

# 
class model():
    def __init__(self):   
        self.label_dict = {}
        self.vector_dict = {}
        df = pd.read_csv('407_dataset.csv')

        # add data from dataset file
        self.df_dict = df.to_dict()
        for i in range(len(self.df_dict['Name'])):
            TN = self.df_dict["Name"][i].split()[0] 
            class_label = self.df_dict["Class"][i]
            if class_label == 'ของใช้ในชีวิตประจำวัน':
                class_label = 'ชอปปิ้ง'
            self.add_new_word(TN,class_label)
        
    def add_new_word(self,new_word,class_label):
        vector = sentence_vectorizer(new_word, use_mean=True) 
        self.label_dict[new_word] = class_label
        self.vector_dict[new_word] = vector

    def name_input(self,TN):
        try: 
            a = self.vector_dict[TN]
            dist_from_1 = {}
#             x = torch.from_numpy(a)
            for key in self.vector_dict:
                y = self.vector_dict[key]
#                 y = torch.from_numpy(self.vector_dict[key])
#                 dist = torch.cosine_similarity(x,y)
                dist = cosine_similarity(a,y)
                dist_from_1[key] = dist
        except:
            print(TN,'not exists')
            a = sentence_vectorizer(TN, use_mean=True) 
            dist_from_1 = {}
#             x = torch.from_numpy(a)
            for key in self.vector_dict:
                y = self.vector_dict[key]
#                 y = torch.from_numpy(self.vector_dict[key])
#                 dist = torch.cosine_similarity(x,y)
                dist = cosine_similarity(a,y)
                dist_from_1[key] = dist
        # KNN        
        k = 10
        knn = []
        sorted_dict = sorted((value,key) for (key,value) in dist_from_1.items())
        for key in sorted_dict[::-1]:
#             print(key)
#             print(self.label_dict[key[1]])
            if (k>=0):
                knn.append(self.label_dict[key[1]])
                k -=1
        class_name = mode(knn)
        if (class_name == 'อาหาร') : class_name = 'food'
        elif (class_name == 'การเดินทาง') : class_name = 'travel'
        elif (class_name == 'สุขภาพ') : class_name = 'health'
        elif (class_name == 'ที่อยู่อาศัย') : class_name = 'resident'
        elif (class_name == 'ของใช้ในครัวเรือน') : class_name = 'household'
        elif (class_name == 'ชอปปิ้ง') : class_name = 'shopping'
        elif (class_name == 'บันเทิง') : class_name = 'entertainment'
        return class_name


app = flask.Flask(__name__)
knn = model()

@app.route('/',methods=['GET'])
def home():
    return 'Hello, This is good wallet api'

@app.route('/classify',methods=['GET'])
def classify():
    transactionName = str(request.args['name'])
    class_name = knn.name_input(transactionName)
    return class_name