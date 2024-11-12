import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import pandas as pd
import pickle
import time


class aesthetic_score:
    def __init__(self, efficientnet_model, model):
        #self.efficientnet_model = EfficientNetV2B0(weights="imagenet", include_top=False, pooling="avg")
        self.efficientnet_model = efficientnet_model
        self.model=model



    def extract_features(self,efficientnet_model,image):
        """Extract features from an image using EfficientNetV2-B0."""
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        
        # Extract features
        features = self.efficientnet_model.predict(image_array)
        return features.flatten()


    def infer_ae_score(self,model, features):# takes the features as input
        features=features.reshape(1,1280)
        try:
            y_pred = self.model.predict(features)
            return y_pred
        except:
            print("failed to predict the score from the model")
            return None
        

    def get_score(self, image):
        start_time=time.time()

        #extract features
        features=self.extract_features(self.efficientnet_model,image)


        #predict aesthetic score
        y_pred_score=self.infer_ae_score(self.model, features)

        print("Aesthetic score of image : ", y_pred_score )

        end_time=time.time()
        print("time taken to infer : ",round((end_time-start_time),4))

        return y_pred_score

    
