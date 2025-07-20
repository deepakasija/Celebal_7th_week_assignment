
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="ML Model Deployment", layout="centered")
st.title(" Iris Species Predictor")
st.write("Input features below to predict the Iris flower species.")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal length (cm)', 'sepal width (cm)',
                                   'petal length (cm)', 'petal width (cm)'])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]

    target_names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f" Prediction: **{target_names[prediction]}**")

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({
        'Species': target_names,
        'Probability': probas
    })
    st.bar_chart(proba_df.set_index('Species'))

st.subheader(" Model Feature Importances")
importances = model.feature_importances_
feat_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

fig, ax = plt.subplots()
sns.barplot(x=importances, y=feat_names, ax=ax)
ax.set_title("Feature Importances")
st.pyplot(fig)