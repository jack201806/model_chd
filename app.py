# Below code is needed for Huggingface

import pathlib
import os
import platform


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
from libreco.algorithms import LightGCN, UserCF, ItemCF, SVD, SVDpp, NCF, NGCF, ALS, BPR
from libreco.evaluation import evaluate


# Models

model_paths = {
    'bpr': './model_path_pure/bpr',
    'svdpp': './model_path_pure/svdpp',
    # extra models paths
}

model_classes = {
    'bpr': BPR,
    'svdpp': SVDpp,
    # extra models
}

def predict(userId, movieId, modelName):
    if modelName not in model_paths:
        raise ValueError("Invalid model name")

    model_path = model_paths[modelName]
    model_used = model_classes[modelName]

    data_info = DataInfo.load(model_path, model_name=modelName+"_model")
    model = model_used.load(path=model_path, model_name=modelName+"_model", data_info=data_info, manual=True)
    prediction = model.predict(user=userId, item=movieId)
    return f"{prediction[0]:.1f}"

def recommend(userId, amount, modelName):
    if modelName not in model_paths:
        raise ValueError("Invalid model name")

    model_path = model_paths[modelName]
    model_used = model_classes[modelName]

    data_info = DataInfo.load(model_path, model_name=modelName+"_model")
    model = model_used.load(path=model_path, model_name=modelName+"_model", data_info=data_info, manual=True)
    recommendations = model.recommend_user(user=userId, n_rec=amount)
    recommended_items = "\n".join(map(str, recommendations[1]))
    return recommended_items

with gr.Blocks() as interfaces:
    with gr.Row():
        model_selection = gr.Dropdown(list(model_paths.keys()), label="Select Model")
    with gr.Row():
        with gr.Column():
            input_userId = gr.Number(label="User ID")
            input_movieId = gr.Number(label="Movie ID")
            input_recommendations = gr.Number(label="How many recommendations?")
        output_pred = gr.Text(label="Predicted rating:")
        output_rec = gr.Text(label="Recommended items:")

    btn_predict = gr.Button(value="Predict", visible=True)
    btn_predict.click(predict, inputs=[input_userId, input_movieId, model_selection], outputs=[output_pred])

    btn_recommend = gr.Button(value="Recommend", visible=True)
    btn_recommend.click(recommend, inputs=[input_userId, input_recommendations, model_selection], outputs=[output_rec])

if __name__ == "__main__":
    interfaces.launch() # share=True, auth=("1", "1")
