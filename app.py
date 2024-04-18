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

# Models

model_paths_rating = {
    'svd_rating': './model_path_pure/svd_rating', # Singular Value Decomposition - RATING
    'svdpp_rating': './model_path_pure/svdpp_rating' # Singular Value Decomposition but includes implicit feedback data - RATING
}

model_paths_ranking = {
    'bpr': './model_path_pure/bpr', # Bayesian Personalized Ranking
    'svd_ranking': './model_path_pure/svd_ranking', # Singular Value Decomposition - RANKING
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

# Dictionary for mapping

item_title_mapping = pd.read_csv("item_title_mapping.csv")
id_to_title = dict(zip(item_title_mapping['item'], item_title_mapping['title']))

# Functions

def predict(userId, movieId, modelName):
    if modelName not in model_paths_rating:
        raise ValueError("Invalid model name")

    model_path = model_paths_rating[modelName]
    model_used = model_classes[modelName]

    data_info = DataInfo.load(model_path, model_name=modelName+"_model")
    model = model_used.load(path=model_path, model_name=modelName+"_model", data_info=data_info, manual=True)
    prediction = model.predict(user=userId, item=movieId)

    return f"{prediction[0]:.1f}"

def predict_by_title(userId, movieTitle, modelName):
    if modelName not in model_paths_rating:
        raise ValueError("Invalid model name")

    movieId = item_title_mapping[item_title_mapping['title'] == movieTitle]['item'].values[0]

    return predict(userId, movieId, modelName)

def recommend(userId, amount, modelName):
    if modelName not in model_paths_ranking:
        raise ValueError("Invalid model name")

    model_path = model_paths_ranking[modelName]
    model_used = model_classes[modelName]

    data_info = DataInfo.load(model_path, model_name=modelName+"_model")
    model = model_used.load(path=model_path, model_name=modelName+"_model", data_info=data_info)

    recommendations = model.recommend_user(user=userId, n_rec=amount)
    recommended_titles = [f"{i+1}. {id_to_title[item_id]}" for i, item_id in enumerate(recommendations[userId])]
    recommended_titles_str = "\n".join(recommended_titles)

    return recommended_titles_str

# Gradio interface

with gr.Blocks() as interface:
    with gr.Tabs() as tab:
        with gr.Tab(label="Predictions"):
            with gr.Row():
                model_selection_pred = gr.Dropdown(list(model_paths_rating.keys()), label="Select Model for Predictions")
            with gr.Row():
                with gr.Column():
                    input_userId_pred = gr.Number(label="User ID", info="Type in the id of the user, e.g. 1")
                    input_movieTitle_pred = gr.Dropdown(choices=item_title_mapping['title'].tolist(), label="Movie Title", filterable=True, info="Start typing a movie title here or select one from the menu.")
                output_pred = gr.Text(label="Predicted rating:", info="This is the predicted rating that the chosen user would give to a movie.")
            btn_predict = gr.Button(value="Predict", visible=True)
            btn_predict.click(predict_by_title, inputs=[input_userId_pred, input_movieTitle_pred, model_selection_pred], outputs=[output_pred])

        with gr.Tab(label="Recommendations"):
            with gr.Row():
                model_selection_rec = gr.Dropdown(list(model_paths_ranking.keys()), label="Select Model for Recommendations")
            with gr.Row():
                with gr.Column():
                    input_userId_rec = gr.Number(label="User ID", info="Type in the id of the user, e.g. 1")
                    input_recommendations_rec = gr.Number(label="How many recommendations?", info="How many movies do you want listed as possible recommendations for the chosen user?")
                output_rec = gr.Text(label="Recommended movies:", info="Here are the top movie recommendations for the chosen user, ranked accordingly")
            btn_recommend = gr.Button(value="Recommend", visible=True)
            btn_recommend.click(recommend, inputs=[input_userId_rec, input_recommendations_rec, model_selection_rec], outputs=[output_rec])

if __name__ == "__main__":
    interface.launch() # share=True, auth=("1", "1")