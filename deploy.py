import streamlit as st  
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model_DenseNet121.h5')

    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

def prediction_cls(prediction): 
    class_names = ['Mild', 'Moderate','Non', 'Very Mild']
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_names[predicted_class_index]
    return predicted_label
    
with st.sidebar:
    st.title("Alzheimer's Detection")
    st.subheader("Accurate detection of Alzheimer's Disease")

st.write("""# Early Prediction of Alzheimer's Disease Detection""")

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    img_array = np.array(image_data)
    img_array = tf.image.resize(img_array, (176, 176))  
    img_array = img_array / 255.0 
    img_array = tf.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)
    probabilities = tf.nn.softmax(prediction[0])
    return probabilities

if file is None:
    st.text("Please upload an MRI Scan file")
else:
    image = Image.open(file)
    image = image.convert("RGB")
    st.image(image ,width =200)
    predictions = import_and_predict(image, model)

    class_names = ['Mild', 'Moderate','Non', 'Very Mild']

    predicted_class = prediction_cls(predictions)

    if predicted_class == 'Very Mild':
        st.write("Your results indicate that you are in the very mild demented stage of Alzheimer's disease.")
        st.sidebar.markdown("## Advise")
        st.sidebar.info("This stage is often characterized by subtle changes in memory and thinking skills. \nIt is crucial to continue monitoring symptoms and seek support from healthcare professionals and caregivers to maintain independence and quality of life.")

    elif predicted_class == 'Mild':
        st.write("Your results suggest that you are in the mild demented stage of Alzheimer's disease.")
        st.sidebar.markdown("## Advise")
        st.sidebar.info("This stage is characterized by mild cognitive decline. \nIt is essential to work closely with your healthcare provider to develop a care plan and explore treatment options to manage symptoms and improve quality of life.")

    elif predicted_class == 'Moderate':
        st.write("Your results suggest that you are in the moderate demented stage of Alzheimer's disease.")
        st.sidebar.markdown("## Advise")
        st.sidebar.info("This stage is characterized by more pronounced cognitive decline and may require more intensive care and support. \nIt is essential to work closely with healthcare providers and caregivers to manage symptoms and provide the best possible quality of life.")

    elif predicted_class == 'Non':
        st.write("Congratulations! Your results indicate that you are in the non-demented stage.")
        st.sidebar.markdown("## Advise")
        st.sidebar.info("This means that there are currently no signs of Alzheimer's disease. \nIt is important to maintain a healthy lifestyle to support brain health and continue regular check-ups with your healthcare provider.")