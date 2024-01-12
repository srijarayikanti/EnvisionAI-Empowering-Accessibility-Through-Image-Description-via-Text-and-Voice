#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from gtts import gTTS
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import json 
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM,Add





# In[5]:


model = load_model("./model_weights/model_9.h5")


# In[6]:


model_temp=ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[7]:


model_resnet=Model(model_temp.input,model_temp.layers[-2].output)


# In[8]:


def preprocess_image(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    return img


# In[9]:


def encode_image(img):
    img=preprocess_image(img)
    feature_vector=model_resnet.predict(img)
    feature_vector=feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector


# In[12]:





# In[15]:





# In[16]:


with open("./storage/word_to_idx.pkl", 'rb') as w2i:
    word_to_idx=pickle.load(w2i)
with open("./storage/idx_to_word.pkl", 'rb') as i2w:
    idx_to_word=pickle.load(i2w)


# In[17]:


def predict_caption(photo):
    in_text = "startseq"
    max_len=35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break
     
    final_caption=in_text.split()
    final_caption=final_caption[1:-1]
    final_caption=' '.join(final_caption)
    language = 'en'
    myobj = gTTS(text=final_caption, lang=language, slow=False)
    if os.path.isfile("./static/audio.mp3"):
        os.remove("./static/audio.mp3")
    for v in range(1000):
        pass
    myobj.save("./static/audio.mp3")
    for v in range(1000):
        pass

    return final_caption


# In[18]:

def caption_this_image(image):
    enc=encode_image(image)
    print("Feature Vector Shape:", enc.shape)
    caption=predict_caption(enc)
    print("Generated Caption:", caption)
    return caption



# In[20]:


from gtts import gTTS
from IPython.display import Audio, display

# Function to convert caption to audio and display an audio button
import tempfile


def caption_to_audio_and_display(image_path):
    enc = encode_image(image_path)
    print("Feature Vector Shape:", enc.shape)
    caption = predict_caption(enc)
    print("Generated Caption:", caption)
    tts = gTTS(text=caption, lang='en')

    # Use a temporary file to save the audio data
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        tts.save(temp_audio_file.name)
        audio_button = Audio(temp_audio_file.name, autoplay=False)
        display(audio_button)

# Use an image defined earlier in your code
image_path = "94232465_a135df2711.jpg"  # Assuming you have defined 'image_path' somewhere earlier

# Convert the caption to audio and display the audio button
caption_to_audio_and_display(image_path)




# In[ ]:




