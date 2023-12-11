import graphene
from fastapi import FastAPI
from starlette.graphql import GraphQLApp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats

# Load a sample dataset for demonstration
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Build a simple neural network using Keras
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

class Query(graphene.ObjectType):
    hello = graphene.String(name=graphene.String(default_value=", World 🌎 !"))
    prediction = graphene.Float(input_data=graphene.List(graphene.Float))

    def resolve_hello(self, info, name):
        return "Hello " + name

    def resolve_prediction(self, info, input_data):
        # Use the trained model for prediction
        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        return prediction[0][0]

app = FastAPI()

app.add_route("/", GraphQLApp(schema=graphene.Schema(query=Query)))
