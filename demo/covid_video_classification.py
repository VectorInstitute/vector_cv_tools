import streamlit as st
import tempfile
import os
from os.path import join
import random

import torch
import numpy as np
import cv2

from model import all_models, get_model
from vector_cv_tools.utils import VideoReader

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

DOCUMENTS_ROOT = os.getenv("CV_DEMO_DOC_ROOT", default="./documents")
MEDIA_FILE_ROOT = join(DOCUMENTS_ROOT, "covid_classification")

CKPT_ROOT = join(MEDIA_FILE_ROOT, "checkpoints")
IMG_ROOT = join(MEDIA_FILE_ROOT, "imgs")
MARKDOWN_ROOT = join(MEDIA_FILE_ROOT, "markdowns")
SAMPLE_ROOT = join(MEDIA_FILE_ROOT, "sample_videos")
TEST_ROOT = join(MEDIA_FILE_ROOT, "test_videos")

sample_videos = [
    "norm1_crop.mp4",
    "norm2_crop.mp4",
    "covid1_crop.mp4",
    "covid2_crop.mp4",
    "pnue1_crop.mp4",
    "pnue2_crop.mp4",
]

test_videos = [
    "normal.mp4",
    "covid.mp4",
    "pneumonia.mp4",
]

sample_video_bytes = [open(join(SAMPLE_ROOT, vid), 'rb').read() for vid in sample_videos]
test_video_bytes = [open(join(TEST_ROOT, vid), 'rb').read() for vid in test_videos]

classes = [
    "Normal/Other",
    "COVID",
    "Pneumonia",
]

loaded_models = {
    "MC3": get_model("MC3_18")(),
    "R3D": get_model("R3D18")(),
}

for name, model in loaded_models.items():
    checkpoint = join(CKPT_ROOT, name + ".ckpt")
    state = torch.load(checkpoint)["model"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

model_card_mc3 = open(join(MARKDOWN_ROOT, "MC3.md")).read()
model_card_r3d = open(join(MARKDOWN_ROOT, "R3D.md")).read()

datacard = open(join(MARKDOWN_ROOT, "datacard.md")).read()
datasource = open(join(MARKDOWN_ROOT, "datasource.md")).read()

desc = open(join(MARKDOWN_ROOT, "covid_intro.md")).read()



def get_video_tensors(path):
    vid = list(VideoReader(path).to_iter())
    dim = (128, 128)
    vid = [ cv2.resize(vid[i], dim, interpolation = cv2.INTER_AREA) \
                for i in range(0, len(vid), 2)]
    vid = np.array(vid)
    vid = torch.from_numpy(vid)
    vid = vid.float() / 255.0
    vid = vid.permute(3, 0, 1, 2)
    vid = vid.unsqueeze(0)
    return vid


test_video_tensors = [get_video_tensors(join(TEST_ROOT, p)) for p in test_videos]
rand_shuffle = random.randint(0, 3)

inference_text = [("", []), ("", []), ("", [])]
ground_truth_res = [False, False, False]

def reset():
    global inference_text
    global ground_truth_res
    inference_text = [("", []), ("", []), ("", [])]
    ground_truth_res = [False, False, False]


def video_classification_page(state):


    st.title("Classification of COVID-19 Based on Lung Ultra-sound")

    # INTRO
    col1, col2 = st.beta_columns(2)
    col1.markdown(desc)
    col2.image(join(IMG_ROOT, "vector_logo.jpg"))

    # Data
    st.markdown(datacard)
    col1, col2 = st.beta_columns([1, 1])
    col1.markdown(datasource)
    col2.markdown("## Conceptual flow of the data collection and processing")
    col2.image(join(IMG_ROOT, "conceptual_flow.png"))

    # Data samples
    example_expander = st.beta_expander("Data Samples")
    cols = [2, 4, 4]
    for i in range(0, len(sample_video_bytes), 2):
        col0, col1, col2 = example_expander.beta_columns(cols)
        col0.markdown("**{}**".format(classes[i // 2]))
        col1.video(sample_video_bytes[i])
        col2.video(sample_video_bytes[i + 1])

    # Model
    st.markdown("# Let's start with selecting a model!")
    models = ("None", "Resnet 3D Model (R3D)",
                      "Mixed Convolutional Network (MC3)")
    selected_model = st.selectbox("", models)

    if len(selected_model) == 0 or selected_model == "None":
        return

    col1, col2 = st.beta_columns([2, 1])
    if "MC3" in selected_model:
        model_card = model_card_mc3
        img_path = join(IMG_ROOT, "mc3.png")
        model_key = "MC3"
    else:
        model_card = model_card_r3d
        img_path = join(IMG_ROOT, "r3d.png")
        model_key = "R3D"

    col1.markdown(model_card)
    col2.image(img_path, width=200, caption="Model Architecture")

    # Live Demo
    demo_expander = st.markdown("# Test the model on real (unseen) videos")
    model_for_inference = loaded_models[model_key]
    demo_expander = st.beta_expander("Test Samples")
    if demo_expander.button("Reset", key="reset"):
        reset()

    cols = [4, 2, 2, 2]
    for i in range(len(test_video_bytes)):
        i = (i + rand_shuffle) % len(test_video_bytes)
        col0, col1, col2, col3 = demo_expander.beta_columns(cols)

        col0.video(test_video_bytes[i])
        col1.markdown("__Take a guess below__")
        user_pred = col1.selectbox("", ["I Don't Know"] + classes,
                                   key="select{}".format(i))
        model_pred = None

        col2.markdown("---")
        if col2.button("Test Video Against Model", key="pred{}".format(i)):
            pred = model_for_inference(test_video_tensors[i].to(device))
            pred_idx = torch.argmax(pred).item()
            beta = 0.5
            pred = pred * beta
            pred = torch.nn.Softmax(dim=0)(pred.flatten()).tolist()

            model_pred = classes[pred_idx]

            prediction_text = ["{:<15}: {:.2f}%".format(cls, prob * 100) \
                                    for cls, prob in zip(classes, pred)]

            inference_text[i] = model_pred, prediction_text

        model_pred, prediction_text = inference_text[i]

        for t in prediction_text:
            col2.write(t)

        if model_pred:
            col2.markdown("\n*__Prediction: {}__*\n".format(model_pred))

        col3.markdown("---")
        if col3.button("Show Ground Truth", key="gt{}".format(i)):
            ground_truth_res[i] = True

        if ground_truth_res[i]:
            ground_truth = classes[i]
            col3.write("Ground Truth:")
            col3.write("__{}__".format(ground_truth))

            col3.markdown("---")
            if model_pred == ground_truth:
                col3.write("Model is correct!!")
            else:
                col3.write("Model is wrong...")

            col3.markdown("---")
            if user_pred == ground_truth:
                col3.write("You are correct!!")
            else:
                col3.write("You are wrong...")
