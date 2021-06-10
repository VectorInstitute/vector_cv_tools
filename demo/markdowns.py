reconstruction_style = """
<style>
    .css-rncmk8 {
        text-align: center;
    }
    .css-1kyxreq {
        display: flex;
        flex-flow: column !important;
        align-items: center;
    }
</style>
"""

background_title = """<h2 align="left">Background</h2>"""

dataset_title = """<h2 align="left">MVTec Dataset</h2>"""

dataset = """
<p align="left">MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects. Pixel-precise annotations of all anomalies are also provided.</p>
<h2 align="center">Overview of Object Categories</h2>
<div class="row">
	<div class="col1">
      <img class="fblogo" src="https://user-images.githubusercontent.com/34798787/120134092-66940380-c19b-11eb-843e-12fc1c2bb692.png" style="width:100%" />
	</div>
	<div class="col2">
        <img class="fblogo" src="https://user-images.githubusercontent.com/34798787/120138284-31d87a00-c1a4-11eb-9016-889815595ce1.png" style="width:110%" />
	</div>
</div>

<style>
    .fblogo {
        display: inline-block;
        margin-left: auto;
        margin-right: auto;
    }
    .row::after {
      content: "";
      clear: both;
      display: table;
      text-align: center;

    }
    .col1 {
      float: left;
      width: 40%;
      margin-left: 70px;
    }
    .col2 {
      float: right;
      width: 40%;
      margin-right: 80px;
    }
</style>
"""

data_anomaly_title = """<h2 align="left">MVTec Dataset for Anomaly Segmentation</h2>"""

data_anomaly = """
<p align="left">The MVTec Dataset consists of both a training and a testing set. The training set contains images of normal objects exclusively. Alternatively, the test set contains images of both normal and defective objects as well as corresponding ground truth masks that localize where the defects are in the image. In the case of anomaly segmentation, the training set is used to model the distribution of normal images and deployed at test time to generate segmentations masks that localize where defects are in the object (if present).</p>
</div>

<style>
    .fblogo {
        display: inline-block;
        margin-left: auto;
        margin-right: auto;
    }
    .row::after {
      content: "";
      clear: both;
      display: table;
      text-align: center;

    }
    .col1 {
      float: left;
      width: 40%;
      margin-left: 70px;
    }
    .col2 {
      float: right;
      width: 40%;
      margin-right: 80px;
    }
</style>
"""

AE_md = """
<div class=model-card-main-div>
	<div class="model-card-div1">
      <h3 align="center">Introduction to Autoencoders</h3>
      <p style="text-align:justify">An autoencoder is an unsupervised neural network that learns to generate low dimensional encodings from which the original input sample can be reconstructed.  As an effective approach to generate compressed representations of data without labels, autoencoders are used widely across different modalities of data including images, videos, text and speech. Autoencoders consist of two key components, an encoder and a decoder. The encoder learns a mapping from an image to a lower-dimensional latent space, and the decoder learns a mapping from the latent space back to the original image. In this way, autoencoders are trained in an unsupervised manner by minimizing the error between the original image and the reconstruction.</p>
        <img src="https://user-images.githubusercontent.com/34798787/120125430-c633e400-c186-11eb-9caf-698f08d017de.png" width="500" />
   </div>

<div class="model-card-div2">
  <h3 align="center">Autoencoders for Anomaly Segmentation in Images</h3>
    <p  style="text-align:justify">As a powerful unsupervised method for learning representations, autoencoders are the basis of many unsupervised anomaly segmentation approaches in image data. In order to do so, the autoencoder is first trained on a set of normal images. At test time, new samples are reconstructed and the pixelwise reconstruction error of a sample is used to identify anomolous pixels by applying a threshold that is determined using a validation set containing both normal and anomalous images. The underlying intuition is that the reconstruction error will be lower for normal pixels than anomalous pixels. This follows from the fact that the autoencoder is trained solely on normal samples so it is unable to reconstruct unseen compositional patterns in anomalous images.</p>
    <img src="https://user-images.githubusercontent.com/34798787/120125362-705f3c00-c186-11eb-82a8-dffa8e2556ff.png" width="500" />
  </div>
</model-card-main-div>

<style>
    .model-card-main-div:after {
      content =""
      clear: both;
      display: table;
    }
    .model-card-div1 {
        text-align:center;
        float: left;
        height: 630px;
        margin-left: 50px;
        width: 600px;
        color: #586069;
        background-color: #fff;
        transition: 0.3s;
        box-shadow: 0 10px 30px -15px rgb(0 0 0 / 20%);
        padding: 2rem;
    }
    .model-card-div2 {
        text-align:center;
        float: right;
        margin-right: 60px;
        width: 600px;
        color: #586069;
        background-color: #fff;
        transition: 0.3s;
        box-shadow: 0 10px 30px -15px rgb(0 0 0 / 20%);
        padding: 2rem;
    }
</style>
"""

VAE_md = """
<div class=model-card-main-div>
	<div class="model-card-div1">
      <h3 align="center">Introduction to Variational Autoencoders</h3>
      <p style="text-align:justify">An autoencoder is an unsupervised neural network that learns to generate low dimensional encodings from which the original input sample can be reconstructed. In contrast to a regular autoencoder, a variational autoenoder enforces that the latent space follow a specified distribution. By ensuring the the latent space is well-structured, variational autoencoders generate more robust representations for downstream tasks than regular autoenoders. As an effective approach to generate compressed representations of data without labels, variational autoencoders are used widely across different modalities of data including images, videos, text and speech. Variational Autoencoders consist of two key components, an encoder and a decoder. The encoder learns a mapping from an image to a lower-dimensional latent space, and the decoder learns a mapping from the latent space back to the original image. In this way, the variational autoencoder is trained in an unsupervised manner by minimizing the error between the original image and the reconstruction as well as enforcing that the latent space follows the specified dsitribution.</p>
        <img src="https://user-images.githubusercontent.com/34798787/120125430-c633e400-c186-11eb-9caf-698f08d017de.png" width="500" />
   </div>

<div class="model-card-div2">
  <h3 align="center">Variational Autoencoders for Anomaly Segmentation in Images</h3>
    <p  style="text-align:justify">As a powerful unsupervised method for learning representations, variational autoencoders are the basis of many unsupervised anomaly segmentation approaches in image data. In order to do so, the autoencoder is first trained on a set of normal images. At test time, the pixelwise reconstruction error of a sample is used to identify anomalous pixels in the image by applying a threshold that is determined using a validation set containing both normal and anomalous images. The underlying intuition is that the reconstruction error will be lower for normal pixels than anomalous pixels. This follows from the fact that the variational autoencoder is trained solely on normal samples so it is unable to reconstruct unseen compositional patterns in anomolous images. </p>
    <img src="https://user-images.githubusercontent.com/34798787/120125362-705f3c00-c186-11eb-82a8-dffa8e2556ff.png" width="500" />
  </div>
</model-card-main-div>

<style>
    .model-card-main-div:after {
      content =""
      clear: both;
      display: table;
    }
    .model-card-div1 {
        text-align:center;
        float: left;
        height: 650px;
        margin-left: 50px;
        width: 600px;
        color: #586069;
        background-color: #fff;
        transition: 0.3s;
        box-shadow: 0 10px 30px -15px rgb(0 0 0 / 20%);
        padding: 2rem;
    }
    .model-card-div2 {
        text-align:center;
        float: right;
        margin-right: 60px;
        width: 600px;
        color: #586069;
        background-color: #fff;
        transition: 0.3s;
        box-shadow: 0 10px 30px -15px rgb(0 0 0 / 20%);
        padding: 2rem;
    }
</style>
"""

CONVVAE_md = """
<div class=model-card-main-div>
	<div class="model-card-div1">
      <h3 align="center">Introduction to Context Autoencoders</h3>
      <p style="text-align:justify">A context autoencoder is an unsupervised neural network that learns to generate low dimensional encodings from which the original input sample can be reconstructed. In contrast to regular autoencoders, context autoencoders learn to reconstruct samples that have had portions of the input sample masked randomly. In this way, context autoencoders offer semantically rich representations by learning to inpaint masked regions of images in addition to reconstructing them. As an effective approach to generate compressed representations of data without labels, context autoencoders are used widely across different modalities of data including images, videos, text and speech. Autoencoders consist of two key components, an encoder and a decoder. The encoder learns a mapping from an image to a lower-dimensional latent space, and the decoder learns a mapping from the latent space back to the original image. In this way, autoencoders are trained in an unsupervised manner by minimizing the error between the original image and the reconstruction; often with a particular emphasis on the masked region of the input samples. </p>
        <img src="https://user-images.githubusercontent.com/34798787/120140024-a52fbb00-c1a7-11eb-990c-c7f993ec3945.png" width="500" />
   </div>

<div class="model-card-div2">
  <h3 align="center">Context Autoencoders for Anomaly Segmentation in Images</h3>
    <p  style="text-align:justify">As a powerful unsupervised method for learning representations, context autoencoders are the basis of many unsupervised anomaly segmentation approaches in image data. In order to do so, the autoencoder is first trained on a set of normal images. At test time, new unmasked samples are reconstructed and the pixelwise reconstruction error of a sample is used to identify anomolous pixels by applying a threshold that is determined using a validation set containing both normal and anomalous images. The underlying intuition is that the reconstruction error will be lower for normal pixels than anomalous pixels. This follows from the fact that the context autoencoder is trained solely on normal samples so it is unable to reconstruct unseen compositional patterns in anomolous images. </p>
    <img src="https://user-images.githubusercontent.com/34798787/120125362-705f3c00-c186-11eb-82a8-dffa8e2556ff.png" width="500" />
  </div>
</model-card-main-div>


<style>
    .model-card-main-div:after {
      content =""
      clear: both;
      display: table;
    }
    .model-card-div1 {
        text-align:center;
        float: left;
        height: 675px;
        margin-left: 50px;
        width: 600px;
        color: #586069;
        background-color: #fff;
        transition: 0.3s;
        box-shadow: 0 10px 30px -15px rgb(0 0 0 / 20%);
        padding: 2rem;
    }
    .model-card-div2 {
        text-align:center;
        float: right;
        margin-right: 60px;
        width: 600px;
        color: #586069;
        background-color: #fff;
        transition: 0.3s;
        box-shadow: 0 10px 30px -15px rgb(0 0 0 / 20%);
        padding: 2rem;
    }
</style>
"""

object_selection_header = """<h2 align="left" style="margin-top: 2rem;">To begin identifying defects, select an object</h2>"""

radio_selection_styles = """
<style>
    div.row-widget.stRadio > div {
        height: 100px;
    }
    div.row-widget.stRadio > div > label {
        margin-bottom: 6px;
    }
</style>"""

selectbox_styles = """
<style>
    div.row-widget.stSelectbox > div {
        margin-bottom: 25px;
    }
</style>
"""

run_model_button = """
<style>
    .css-qbe2hs {
        padding: 1rem 1.5rem;
        background-color: rgb(246, 51, 102);
        color: white;
        margin: 60px 0px;
        font-size:
        border: solid 1px rgb(246, 51, 102);
        width: 200px;
        font-size: 18px;
    }
    .css-qbe2hs:hover {
        font-weight: 600;
        box-shadow: rgba(0, 0, 0, 0.2) 0px 10px 30px -15px;
        transition: all .3 ease 0s;
        color: white;
    }
    .css-qbe2hs:focus:not(:active) {
        color: white;
    }

</style>
"""

references = """## References
* Bergmann, Paul, et al. "MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
* Baur, Christoph, et al. "Autoencoders for unsupervised anomaly segmentation in brain mr images: A comparative study." Medical Image Analysis (2021): 101952.
"""

sys_intro = """## Infrastructure and Technical Support

The demo leverages the unique resources and capabilities at Vector Institute
ranging from project management, infrastructure, tooling and deployment.

"""

acknowledgements = """## Thanks to everyone who has made this demo possible

**Computer Vision Project Participants and Vector Sponsors**

**Vector Industry Team**

**Vector Researchers**

**Vector AI Engineering Team**

**Vector Scientific Computing Team**

"""

