# ThumbnailAI

This application processes a video and suggests the best thumbnail to use. The selection of best thumbnail is done by selecting the most aesthetic video frame in the given video. To determine the aesthetic score an EfficientV2 B0 model without its head is used to extract the image features, which is fed into a Xgboost model to determine the Aesthetic score. The given ML model is trained on FLICKRS-AES image dataset. I have found the preprocessed data at [link](https://github.com/alanspike/personalizedImageAesthetics).
