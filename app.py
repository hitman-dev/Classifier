import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing import image
from keras.applications.xception import preprocess_input

# streamlit run app.py --server.address=127.0.0.1

st.set_page_config(layout="wide")


def load_image(image_file):
    img = Image.open(image_file)
    return img


def classify_monkey(image):

    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, 299, 299, 3))

    # prepare the image for the VGG model
    image = preprocess_input(image)
    classes = monkey_model.predict(image)
    max_arg = np.argmax(classes)
    return df["Common Name"].iloc[max_arg]


def classify_room(img):
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(
        img,
        (224, 224),
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=True,
    )
    img = np.array(img)
    img = img.reshape(1, 224, 224, 3)
    prob = room_model.predict(img)
    if prob[0] > 0.5:
        return "Messy Room", " This room is Messy 🤢. Please Clean it Up."
    else:
        return "Clean Room", " This room is Clean 🤩. Good Work."


df = pd.read_csv(
    "Monkey/monkey_labels.txt"
)


@st.cache(allow_output_mutation=True)
def load_ml_model(path):
    return tf.keras.models.load_model(path)


room_model = load_ml_model(
    "CleanRoom_MessyRoom/clean-messy_model"
)

with open('Monkey/models/monkey_model.pkl' , 'rb') as f:
    monkey_model = pickle.load(f)


menu = ["Monkey Species", "Clean Room or Messy Room", "Fruits", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Monkey Species":
    st.header("Monkey Classifier")
    st.markdown(
        "#### Here are the 10 monkey species. This classifier will classify the uploaded image according to its species."
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    counter = 0
    for col in st.columns(5):
        with col:
            name = df["Common Name"].iloc[counter]
            st.text(name)
            path = f"Monkey/monkey images/{name}.jpg"
            image = Image.open(path)
            st.image(image)
            with st.expander("Details"):
                html_str = f"""
                <h5>{name}</h5> <p>{df['Details'].iloc[counter]}</p> 
                """
                st.markdown(html_str, unsafe_allow_html=True)
        counter += 1
    counter = 5
    for col in st.columns(5):
        with col:
            name = df["Common Name"].iloc[counter]
            st.text(name)
            path = f"Monkey/monkey images/{name}.jpg"
            image = Image.open(path)
            st.image(image)
            with st.expander("Details"):
                html_str = f"""
                <h5>{name}</h5> <p>{df['Details'].iloc[counter]}</p> 
                """
                st.markdown(html_str, unsafe_allow_html=True)
        counter += 1

    image_file = st.file_uploader("Upload Image of Monkey", type=["png", "jpg", "jpeg"])
    if st.button("Predict Species"):
        if image_file is not None:
            # To View Uploaded Image
            col1, col2 = st.columns([1, 3])
            with col1:
                img = load_image(image_file)
                st.image(img)
                newsize = (299, 299)
                img = img.resize(newsize)
            with col2:
                name = classify_monkey(img)
                details = df.loc[df["Common Name"] == name, "Details"].iloc[0]
                html_str = f"""
                <h2>{name}</h2> <p style="font-size:20px">{details}</p> 
                """
                st.markdown(html_str, unsafe_allow_html=True)
        else:
            st.warning("Image not Uploaded!!")

elif choice == "Clean Room or Messy Room":
    st.header("Clean Room Messy Room Classifier")
    st.markdown(
        "#### If you upload an image of a room, then this classifier will tell you if your room is Clean or Messy."
    )
    image_file = st.file_uploader(
        "Upload Image of the Room", type=["png", "jpg", "jpeg"]
    )

    if st.button("Predict"):
        if image_file is not None:
            # To View Uploaded Image
            col1, col2 = st.columns([1, 3])
            with col1:
                img = load_image(image_file)
                st.image(img)
                newsize = (224, 224)
                img = img.resize(newsize)
            with col2:
                room_condition, sentense = classify_room(img)
                html_str = f"""
                <h2>{room_condition}</h2> <p style="font-size:20px">{sentense}</p> 
                """
                st.markdown(html_str, unsafe_allow_html=True)
