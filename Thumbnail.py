
import cv2
from PIL import Image
from Aesthetic_Score_xgboost import aesthetic_score
import time
from multiprocessing import Pool
from tensorflow.keras.applications import EfficientNetV2B0
import pickle

# Helper function to initialize models in each worker
def init_model():
    global efficientnet_model, xgb_model
    efficientnet_model = EfficientNetV2B0(weights="imagenet", include_top=False, pooling="avg")
    with open("xgb_model-full.pkl", "rb") as f:
        xgb_model = pickle.load(f)

# Process each frame and compute aesthetic score
def process_frame(frame):
    ae = aesthetic_score(efficientnet_model, xgb_model)
    start_time = time.time()
    frame_data = {
        'frame': frame,
        'score': ae.get_score(Image.fromarray(frame))
    }
    end_time = time.time()
    print("Time taken to process this frame (in minutes):", round((end_time - start_time) / 60, 2))
    return frame_data

if __name__ == '__main__':
    print("Starting the App")
    start_time = time.time()
    
    # Load video and retrieve metadata
    cap = cv2.VideoCapture("INSERT VIDEO PATH")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"FPS of video: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}")

    # Extract frames at intervals
    frame_list = []
    for count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % fps == 0:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)

    print("Frame list extracted.")
    print(f"The length of frame list is: {len(frame_list)}")
    # Release video capture resource
    cap.release()
    
    # Initialize model in each process and process frames
    with Pool(processes=6, initializer=init_model) as pool:
        scores_list = pool.map(process_frame, frame_list)

    # Sort the results by score
    sorted_scores_list = sorted(scores_list, key=lambda x: x['score'], reverse=True)
    print(sorted_scores_list[:3])
    print(len(sorted_scores_list))
    

    
    end_time = time.time()
    print("Total time taken to process video (in minutes):", round((end_time - start_time) / 60, 2))

