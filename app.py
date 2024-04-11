# Below code is needed for Huggingface

import pathlib
import os

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

dir_path = os.path.dirname(os.path.realpath(__file__))

# Imports

import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
import gradio as gr

from libreco.data import random_split, DatasetPure, DataInfo
from libreco.algorithms import LightGCN, UserCF, ItemCF, SVD, SVDpp, NCF, NGCF, ALS
from libreco.evaluation import evaluate

data_info = DataInfo.load("model_path", model_name="svdpp_model")
model = SVDpp.load(
        path=dir_path+"/model_path", model_name="svdpp_model", data_info=data_info, manual=True
    )

# Functions

def predict(userId, movieId):
    prediction = model.predict(user=userId, item=movieId)
    return f"{prediction[0]:.1f}"

def recommend(userId, amount):
    recommendations = model.recommend_user(user=userId, n_rec=amount)
    recommended_items = "\n".join(map(str, recommendations[1]))
    return recommended_items

with gr.Blocks() as interfaces:
    with gr.Row():
        with gr.Column():
            input_userId = gr.Number(label="User ID")
            input_movieId = gr.Number(label="Movie ID")
            input_rating = gr.Number(label="How many recommendations?")
        output_pred = gr.Text(label="Predicted rating:")
        output_rec = gr.Text(label="Recommended items:")

    btn_predict = gr.Button(value="Predict")
    btn_predict.click(predict, inputs=[input_userId, input_movieId], outputs=[output_pred])

    btn_recommend = gr.Button(value="Recommend")
    btn_recommend.click(recommend, inputs=[input_userId, input_rating], outputs=[output_rec])

if __name__ == "__main__":
    interfaces.launch()