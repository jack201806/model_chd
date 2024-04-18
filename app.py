# Huggingface code

import pathlib
import os
import platform


plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

dir_path = os.path.dirname(os.path.realpath(__file__))

# Imports

import pandas as pd
import gradio as gr

from libreco.data import DataInfo
from libreco.algorithms import LightGCN, SVD, SVDpp, NGCF, BPR, DeepWalk

# Dictionary for mapping
item_title_mapping = pd.read_csv("item_title_mapping.csv")
id_to_title = dict(zip(item_title_mapping['item'], item_title_mapping['title']))

# Models

model_paths = {
    'bpr': './model_path_pure/bpr', # Bayesian Personalized Ranking
    'svd_rating': './model_path_pure/svd_rating', # Singular Value Decomposition - RATING
    'svd_ranking': './model_path_pure/svd_ranking', # Singular Value Decomposition - RANKING
    'svdpp_rating': './model_path_pure/svdpp_rating', # Singular Value Decomposition but includes implicit feedback data - RATING
    'svdpp_ranking': './model_path_pure/svdpp_ranking', # Singular Value Decomposition but includes implicit feedback data - RANKING
    'ngcf': './model_path_pure/ngcf', # Neural Graph Collaborative Filtering
    'lightgcn': './model_path_pure/lightgcn', # Simplified, more scalable version of NGCF
    'deepwalk': './model_path_pure/deepwalk'
}

model_classes = {
    'bpr': BPR,
    'svd_rating': SVD,
    'svd_ranking': SVD,
    'svdpp_rating': SVDpp,
    'svdpp_ranking': SVDpp,
    'ngcf': NGCF,
    'lightgcn': LightGCN,
    'deepwalk': DeepWalk
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
    model = model_used.load(path=model_path, model_name=modelName+"_model", data_info=data_info)

    recommendations = model.recommend_user(user=userId, n_rec=amount)
    recommended_titles = [id_to_title[item_id] for item_id in recommendations[1]]
    recommended_titles_str = "\n".join(str(title) for title in recommended_titles)

    return recommended_titles_str

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