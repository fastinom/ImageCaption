import streamlit as st 
from PIL import Image
from os.path import join, dirname, realpath
from glob import glob
import numpy as np
import os
import cv2,imutils
import tensorflow as tf

from transformers import ViTImageProcessor

import pickle

st.write('Fastino Mateteva')

# TODO : Load the tokenizer.pickle in this directory object file
with open('tokenizer.pickle','rb') as tokenizer_file:
    TOKENIZER = pickle.load(tokenizer_file)

# The model to generate the caption
# TODO : Load the caption_model.h5 file in this directory
CAPTIONMODEL =  tf.keras.models.load_model('caption_model.h5')
# Feature extractor
# TODO : Load the feature_extractor.h5 in this directory
# IMAGEFEATUREEXTRACTOR = tf.keras.models.load_model('feature_extractor.h5')
IMAGEFEATUREEXTRACTOR = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Frames path
FRAMES = join(dirname(realpath(__file__)), "frames")
 
def load_image(image_file):
	img = Image.open(image_file)
	return img

#  upload video
video = st.file_uploader(label="upload video", type="mp4", key="video_upload_file")

# Continue only if video is uploaded successfully
if(video is not None):
    if os.path.exists(FRAMES):
        frame_paths = glob(f"frames/*.jpeg")
        for path in frame_paths:
            os.remove(path)
        os.rmdir(FRAMES)
    # Notify user
    st.text("Video has been uploaded")
    # Gather video meta data
    file_details = {
        "filename":video.name, 
        "filetype":video.type,
        "filesize":video.size
    }
    # Show on ui
    st.write(file_details)
    # save video
    with open(video.name, "wb") as f:
        f.write(video.getbuffer())
    
    st.success("Video saved")

    video_file = open(file_details['filename'], 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    def create_frames():

        # Create frames directory
        os.makedirs(FRAMES)

        images_array = []

        cap = cv2.VideoCapture(video.name)
        index = 0

        while True:
            ret, frame = cap.read()
            if ret == False:
                cap.release()
                break
            if frame is None:
                break
            else:
                if index == 0:
                    images_array.append(frame)
                    cv2.imwrite(f"frames/{index}.jpeg", frame)

                else:
                    if index % 10 == 0:
                        images_array.append(frame)
                    cv2.imwrite(f"frames/{index}.jpeg", frame)

            index += 1
        return np.array(images_array)

    images_array = create_frames()

    def idx_to_word(integer,tokenizer):
        for word, index in tokenizer.word_index.items():
            if index==integer:
                return word
        return None

    def predict_caption(image_path, max_length=34):
        model = tf.keras.applications.DenseNet201()
        fe = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

        img_size = 224
        features = {}

        img = tf.keras.preprocessing.image.load_img(image_path,target_size=(img_size,img_size))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img/255.
        img = np.expand_dims(img,axis=0)
        feature = fe.predict(img, verbose=0)
        features[image_path] = feature

        
        feature = features[image_path]
        in_text = "startseq"

        for i in range(max_length):
            sequence = TOKENIZER.texts_to_sequences([in_text])[0]
            sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], max_length)

            y_pred = CAPTIONMODEL.predict([feature,sequence])
            y_pred = np.argmax(y_pred)
            
            word = idx_to_word(y_pred, TOKENIZER)
            
            if word is None:
                break
                
            in_text+= " " + word
            
            if word == 'endseq':
                break
                
        return in_text

    if len(images_array) > 0:
        frame_paths = glob(f"frames/*.jpeg")
        for path in frame_paths:
            caption = predict_caption(path)
            st.image(load_image(path), caption=caption, width=250)
