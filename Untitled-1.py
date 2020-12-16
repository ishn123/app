import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm
import hashlib
from database_connection import *

def hashing(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    img = np.expand_dims(img_array,0)
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    last_conv_layer_output = last_conv_layer_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    #heatmap = cv2.resize(heatmap,(224,224),cv2.INTER_LINEAR)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def superimposed_img(image, heatmap):
    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((224,224))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def frontend(gui = True):
    if gui:
        im1 = Image.open('random.PNG')
        st.title('Swasthya setu')
        st.markdown('''A Deep Learning Application used to Analyze the presence of **_COVID-19_** in Patients. Healthcare sector is totally different from other industry. 
                It is on high priority sector and people expect highest level of care and services regardless of cost. 
                It did not achieve social expectation even though it consume huge percentage of budget. 
                Mostly the interpretations of medical data is being done by medical expert. In terms of image interpretation by
                human expert, it is quite limited due to its subjectivity, complexity of the image, extensive variations exist across different interpreters, and fatigue.
                After the success of deep learning in other real world application, it is also
                providing exciting solutions with good accuracy for medical imaging and is
                seen as a key method for future applications in health sector. In this
                chapter, we discussed state of the art deep learning architecture and its
                optimization used for medical image segmentation and classification.''')
        st.image([im1],width=600)
        st.write('\n')
        st.subheader('Steps to upload your X-ray')
        st.markdown('1. Go to the side bar and Login/SignUp.\n 2. After Login in Upload your X-ray.')
        option = st.sidebar.selectbox(label='Menu',options=['Create account','Login'],)
        if option =='Create account':
            create_user()
            first_name = st.sidebar.text_input(label='First Name')
            last_name = st.sidebar.text_input(label='Last Name')
            Emailaddress = st.sidebar.text_input(label='Email Address')
            if Emailaddress != '':
                uniqueess_checker(Emailaddress,'emailaddress')
            Username = st.sidebar.text_input(label='Username')
            if Username != '':
                uniqueess_checker(Username,'username')
            Password = st.sidebar.text_input(label='Password',type='password')
            Signup = st.sidebar.button('Signup')
            if Signup:
                encrypt = hashing(Password)
                add_user_data(first_name,last_name,Emailaddress,Username,encrypt)
                st.success('Account created successfully.Now go to login to upload your X-ray.')
        else:
            username = st.sidebar.text_input(label='Username',key='Username')
            password = st.sidebar.text_input(label='password',key='password',type='password')
            login = st.sidebar.checkbox(label='Login')
            if login:
                hshd_pss = hashing(password)
                rows = two_columns_retrieval(username,hshd_pss)
                if rows !=[]:
                    st.success('Login successfully as {}'.format(username))
                    st.write('Now you can upload your X-ray.')
                    prediction()
                else:
                    st.warning('Username/Password is Incorrect')
        
def uniqueess_checker(column_argument,column):
    col = column_data(column)
    for a in col:
        if column_argument == a[0]:
            if column == 'emailaddress':
                st.sidebar.warning('Emailaddress already exists,Try Login.')
                break
            else:
                st.sidebar.warning('Username already exists,Try with diiferent one')
                break

def prediction():

    upload = st.file_uploader(label='',type = ['png','jpg','jpeg'])
    analyze = st.button('Analyze report')
    if analyze:
        cnn_model = keras.models.load_model('my_model.h5')
        with st.spinner('Wait for a while'):
            img = Image.open(upload,'r')
            img = img.resize((224,224))
            img = np.array(img)
            img = img.reshape(224,224,3)
            img = img/255
            pred = cnn_model.predict(np.expand_dims(img,0))
            if np.argmax(pred) == 1:
                st.write('You have been diagonsed with **_COVID-19 !_** Please consult Doctor immediately.')
                heatmap=make_gradcam_heatmap(img,cnn_model,last_conv_layer_name='conv_pw_13_relu',classifier_layer_names=['global_average_pooling2d','dense_1'])
                ima = np.uint8(255 * img)
                s_img = superimposed_img(ima,heatmap)
                s_img.save('img1.png')
                display_img = Image.open('img1.png','r')
                st.image(display_img,width = 150,caption = st.write('**_Red_** marks in X-ray indicates swelling in Bronchitis.'))
            else:
                st.write('Your X-ray is **_Normal_**.')
if __name__ == '__main__':
    frontend()
