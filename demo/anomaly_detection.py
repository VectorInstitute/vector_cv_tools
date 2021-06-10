import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from vector_cv_tools import transforms as VT
from vector_cv_tools import datasets as vdatasets
from vector_cv_tools.datasets.mvtec import MVTec_OBJECTS

import os
from os.path import join

import streamlit as st
import fishae
import fishvae
import convvae

import markdowns

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

DOCUMENTS_ROOT = os.getenv("CV_DEMO_DOC_ROOT", default="./documents")
MEDIA_FILE_ROOT = join(DOCUMENTS_ROOT, "anomaly_detection")
MVTEC_ROOT_DIR = os.path.join(MEDIA_FILE_ROOT, "dataset/MVTec_AD")
CKPT_ROOT_DIR = os.path.join(MEDIA_FILE_ROOT, "models")

basic_transform = VT.ComposeMVTecTransform(
    [A.Resize(128, 128),
     A.ToFloat(max_value=255),
     ToTensorV2()])

dsets = list(
    vdatasets.MVTec(MVTEC_ROOT_DIR,
                    split="test",
                    transforms=basic_transform,
                    obj_types=[obj_type]) \
    for obj_type in MVTec_OBJECTS)

img_data = [
    join(MVTEC_ROOT_DIR, "bottle/test/broken_large/004.png"),
    join(MVTEC_ROOT_DIR, "tile/test/crack/002.png"),
    join(MVTEC_ROOT_DIR, "screw/test/manipulated_front/003.png"),
    join(MVTEC_ROOT_DIR, "zipper/test/broken_teeth/001.png"),
    join(MVTEC_ROOT_DIR, "toothbrush/test/defective/000.png"),
    join(MVTEC_ROOT_DIR, "metal_nut/test/bent/000.png")
]

imgs = [open(i, 'rb').read() for i in img_data]

loaded_images = {
    "Bottle": imgs[0],
    "Tile": imgs[1],
    "Screw": imgs[2],
    "Zipper": imgs[3],
    "Toothbrush": imgs[4],
    "Metal Nut": imgs[5],
}

loaded_examples = {
    "Bottle": dsets[0][4],
    "Tile": dsets[10][2],
    "Screw": dsets[9][44],
    "Zipper": dsets[14][1],
    "Toothbrush": dsets[11][0],
    "Metal Nut": dsets[7][0],
}

loaded_examples = {k : \
                       [item[0].unsqueeze(0),
                        item[1]['mask'].unsqueeze(0).unsqueeze(0)]\
                       for k, item in loaded_examples.items()}


def imgs2vecs(model, img):
    with torch.no_grad():
        z = model.module.encoder(img)
    return z[..., :100]


def vecs2imgs(model, vec):
    with torch.no_grad():
        img = model.module.decoder(vec.unsqueeze(-1).unsqueeze(-1))
    return img


def imgs2imgs(model, img):
    return vecs2imgs(model, imgs2vecs(model, img))


def show_reconstruction(*imgs):
    plt.rcParams["figure.figsize"] = (3 * len(imgs), 3)
    col1, original_image_col, reconstruction_col, col4 = st.beta_columns(
        [1, 2, 2, 1])
    st.markdown(markdowns.reconstruction_style, unsafe_allow_html=True)
    original_single_img = imgs[0].permute(1, 2, 0).cpu().numpy()
    rec_single_img = imgs[1].permute(1, 2, 0).cpu().numpy()
    original_image_col.subheader('Original Image')
    original_image_col.image(original_single_img)
    reconstruction_col.subheader('Reconstructed Image')
    reconstruction_col.image(rec_single_img)


def show_threshold_img(img):
    threshold_single_img = img.permute(1, 2, 0).cpu().numpy()
    st.markdown(markdowns.reconstruction_style, unsafe_allow_html=True)
    col1, col2, col3 = st.beta_columns([1, 1, 1])
    col2.subheader('Threshold Image')
    col2.image(threshold_single_img)


def show_rec_dif(model, img, mask):
    rec = imgs2imgs(model, img.to(device))
    dif_raw = abs(img.to(device) - rec)
    dif = (dif_raw - dif_raw.min()) / (dif_raw.max() - dif_raw.min())
    show_reconstruction(img.squeeze(), rec.squeeze())


def show_threshold_dif(model, img, mask, threshold):
    rec = imgs2imgs(model, img.to(device))
    dif_raw = abs(img.to(device) - rec)
    dif = (dif_raw - dif_raw.min()) / (dif_raw.max() - dif_raw.min())
    selected_threshold = (dif + threshold).round().squeeze()
    show_threshold_img(selected_threshold)


aeModel = torch.nn.DataParallel(fishae.FishAE())
aeModel.load_state_dict(torch.load(join(CKPT_ROOT_DIR, "fishae.pt")))
aeModel.eval()
vaeModel = torch.nn.DataParallel(fishvae.FishVAE())
vaeModel.load_state_dict(torch.load(join(CKPT_ROOT_DIR, "fishvae.pt")))
vaeModel.eval()
convvaeModel = torch.nn.DataParallel(convvae.ConvVAE())
convvaeModel.load_state_dict(torch.load(join(CKPT_ROOT_DIR, "context_vae.pt")))
convvaeModel.eval()

loaded_models = {
    "Regular Autoencoder": aeModel,
    "Variational Autoencoder": vaeModel,
    "Context Autoencoder": convvaeModel
}

show_rec = False


def anomaly_detection_page(state):
    col1, col2, col3 = st.beta_columns([1, 2, 1])
    col2.title("Anomaly Detection in Manufacturing")

    model_cards = {
        "Regular Autoencoder": markdowns.AE_md,
        "Variational Autoencoder": markdowns.VAE_md,
        "Context Autoencoder": markdowns.CONVVAE_md
    }

    st.markdown(markdowns.background_title, unsafe_allow_html=True)
    with st.beta_expander("See more"):
        bcol1, bcol2, bcol3, bcol4, bcol5 = st.beta_columns([.2, 2, .8, 2, .01])
        bcol2.image("assets/machine_parts.png", width=550)
        bcol4.image("assets/fish.png", width=350)
        bcol6, bcol7, bcol8 = st.beta_columns([1, 2.6, 1.2])
        bcol7.image("assets/defect_alert.png")

    st.markdown(markdowns.dataset_title, unsafe_allow_html=True)
    with st.beta_expander("See more"):
        st.markdown(markdowns.dataset, unsafe_allow_html=True)

    st.markdown(markdowns.data_anomaly_title, unsafe_allow_html=True)
    with st.beta_expander("See more"):
        st.markdown(markdowns.data_anomaly, unsafe_allow_html=True)
        acol1, acol2, acol3, acol4, acol5 = st.beta_columns(
            [.3, 1, .1, 1.3, .2])
        acol2.image("assets/ad_collection.png")
        acol4.image("assets/hazelnut.png")

    st.markdown(markdowns.object_selection_header, unsafe_allow_html=True)

    radio_selection_col, image_selection_col, col3 = st.beta_columns([2, 1, 1])
    with radio_selection_col:
        st.write(markdowns.radio_selection_styles, unsafe_allow_html=True)
        selected_image = st.radio("", list(loaded_images.keys()))

    with image_selection_col:
        st.image(loaded_images[selected_image],
                 caption='Selected Object',
                 width=125)

    img, mask = loaded_examples[selected_image]

    model_options = ("None", "Regular Autoencoder", "Variational Autoencoder",
                     "Context Autoencoder")
    st.write(markdowns.selectbox_styles, unsafe_allow_html=True)
    selected_model = st.selectbox("Select a model", model_options)

    if len(selected_model) == 0 or selected_model == "None":
        return

    model = loaded_models[selected_model]
    st.markdown(model_cards[selected_model], unsafe_allow_html=True)
    col1, col2, col3 = st.beta_columns([1, 1, 1])

    st.markdown(markdowns.run_model_button, unsafe_allow_html=True)
    global show_rec
    if col2.button("Run Model"):
        show_rec = True

    if show_rec:
        show_rec_dif(model, img, mask)

    threshold = st.slider("Threshold Value",
                          min_value=-0.3,
                          max_value=0.3,
                          value=0.0,
                          step=0.001)
    if threshold:
        show_threshold_dif(model, img, mask, threshold)

    st.markdown(markdowns.references, unsafe_allow_html=True)
