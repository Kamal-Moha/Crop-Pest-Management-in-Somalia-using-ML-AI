import streamlit as st
from PIL import Image
import fastai
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
#from pathlib import Path
import joblib
from time import sleep
import warnings
warnings.filterwarnings('ignore')

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

st.header('Detect The Disease Affecting your Crop', divider='rainbow')
st.markdown('Simply upload a picture of your affected crop and our system will tell what is affecting it')

maize = 'new_maize_fastai_model.pkl'
cassava = 'new_cassava_fastai_model.pkl'
tomato_crop = 'Tomato_Crop_fastai_model.pkl'

# Choosing the crop
crop_option = st.selectbox('What Crop do you want to Detect?', ('Maize', 'Cassava', 'Tomato Crop'), index=None)

# creating the path file according to the crop_option
# file = maize if crop_option == 'Maize' else cassava if crop_option == 'Cassava'
if crop_option == 'Maize':
  file = maize
elif crop_option == 'Cassava':
  file = cassava
elif crop_option == 'Tomato Crop':
  file = tomato_crop

# Upload file
uploaded_file = st.file_uploader('Upload Image', type=['png', 'jpg'])

if uploaded_file is not None:

  # To read file as bytes:
  file_img = st.image(uploaded_file)

  img = PILImage.create(uploaded_file)
  if st.button('Detect Pest'):
    # model = load_learner(Path('E:\\Omdena Projects\\Crop Pest Management in Somalia using AI\\new_maize_fastai_model.pkl'))
    model = load_learner(file)
    model.remove_cb(ProgressCallback)

    with st.spinner('Wait for it...'):
        sleep(5)
    pre_class, pre_ind, output = model.predict(img)
    st.write(f"What's this?: {pre_class}.")
    st.write(f"Probability it's a {pre_class}: {output[pre_ind].item():.6f}")


    # Creating a dictionary matching each category to its probability
    res = {model.dls.vocab[i]: float(output[i]) for i in range(len(output))}

    # sorting the dictionary by values in descending order
    sorted_res = sorted(res.items(), key=lambda x:x[1], reverse=True)


    for i in sorted_res:
      st.progress(i[1], text=i[0])

