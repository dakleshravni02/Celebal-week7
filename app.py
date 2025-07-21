import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌼 Iris Flower Classifier")

st.write("Input the features below:")

# Input form
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=["sepal length (cm)", "sepal width (cm)", 
                                       "petal length (cm)", "petal width (cm)"])
    prediction = model.predict(input_data)[0]
    flower_names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Iris Species: **{flower_names[prediction]}**")
