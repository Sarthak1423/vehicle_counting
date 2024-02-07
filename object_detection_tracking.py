import streamlit as st
st.set_page_config(page_title='Vehicle Counting', page_icon='🚕', layout="wide") 

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import warnings
import torch
from ultralytics import YOLO
from collections import deque

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

from helper import create_video_writer
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = '0' if torch.cuda.is_available() else 'cpu'
save_path = "video.mp4"
output_video_path = 'output.avi'
final_video_path = 'output.mp4'
object_list = ['car', 'truck', 'motorbike', 'bus']
model = YOLO("model/yolov8m.pt")


def convert_avi_to_mp4(avi_file_path, output_name):
    os.system(f"ffmpeg -i {avi_file_path} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output_name}")
    os.remove(avi_file_path)
    return True

def video_process():
    # define some parameters
    conf_threshold = 0.5
    max_cosine_distance = 0.4
    nn_budget = None
    points = [deque(maxlen=32) for _ in range(1000)] # list of deques to store the points
    counter_A = 0
    counter_B = 0
    counter_C = 0
    start_line_A = (0, 480)
    end_line_A = (480, 480)
    start_line_B = (525, 480)
    end_line_B = (745, 480)
    start_line_C = (895, 480)
    end_line_C = (1165, 480)
    
    video_cap = cv2.VideoCapture(save_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = create_video_writer(video_cap, output_video_path)

    # Initialize the deep sort tracker
    model_filename = "config/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # load the COCO class labels the YOLO model was trained on
    classes_path = "config/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    print(class_names)
    # create a list of random colors to represent each class
    np.random.seed(42)  # to get the same colors
    colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

    progress_text = "Processing..." 
    st.write("## Real-time process: ")
    my_bar = st.progress(0, text=progress_text)
    # loop over the frames
    trackid_dictionary = {}
    empty_container = st.empty()

    while video_cap.isOpened():
        current_frame = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        my_bar.progress(int((current_frame/total_frames)*100), text=progress_text)

        ret, frame = video_cap.read()

        if not ret:
            print("End of the video file...")
            break

        overlay = frame.copy()
        # draw the lines
        cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 12)
        cv2.line(frame, start_line_B, end_line_B, (255, 0, 0), 12)
        cv2.line(frame, start_line_C, end_line_C, (0, 0, 255), 12)
        
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        results = model.predict(frame, save=False, device=device)
        # results = model(frame)

        # loop over the results
        for result in results:
            # initialize the list of bounding boxes, confidences, and class IDs
            bboxes = []
            confidences = []
            class_ids = []

            # loop over the detections
            for data in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = data
                x = int(x1)
                y = int(y1)
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)
                class_id = int(class_id)

                # filter out weak predictions by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > conf_threshold:
                    bboxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
        ############################################################
        ### Track the objects in the frame using DeepSort        ###
        ############################################################

        # get the names of the detected objects
        names = [class_names[class_id] for class_id in class_ids]

        # get the features of the detected objects
        features = encoder(frame, bboxes)
        # convert the detections to deep sort format
        dets = []
        for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
            dets.append(Detection(bbox, conf, class_name, feature))

        # run the tracker on the detections
        tracker.predict()
        tracker.update(dets)

        # loop over the tracked objects
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # get the bounding box of the object, the name
            # of the object, and the track id
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_name = track.get_class()
            # convert the bounding box to integers
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # get the color associated with the class name
            class_id = class_names.index(class_name)
            color = colors[class_id]
            B, G, R = int(color[0]), int(color[1]), int(color[2])

            # draw the bounding box of the object, the name
            # of the predicted object, and the track id
            
            if class_name in object_list:
                if class_name in trackid_dictionary:
                    trackid_dictionary[class_name].append(track_id)
                else:
                    trackid_dictionary[class_name] = [track_id]

                text = str(track_id) + " - " + class_name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 3)
                cv2.rectangle(frame, (x1 - 1, y1 - 20),
                            (x1 + len(text) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                ############################################################
                ### Count the number of vehicles passing the lines       ###
                ############################################################
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # append the center point of the current object to the points list
                points[track_id].append((center_x, center_y))

                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
                
                # loop over the set of tracked points and draw them
                for i in range(1, len(points[track_id])):
                    point1 = points[track_id][i - 1]
                    point2 = points[track_id][i]
                    # if the previous point or the current point is None, do nothing
                    if point1 is None or point2 is None:
                        continue
                    
                    cv2.line(frame, (point1), (point2), (0, 255, 0), 2)
                    
                # get the last point from the points list and draw it
                last_point_x = points[track_id][0][0]
                last_point_y = points[track_id][0][1]
                cv2.circle(frame, (int(last_point_x), int(last_point_y)), 4, (255, 0, 255), -1)    

                # if the y coordinate of the center point is below the line, and the x coordinate is 
                # between the start and end points of the line, and the the last point is above the line,
                # increment the total number of cars crossing the line and remove the center points from the list
                if center_y > start_line_A[1] and start_line_A[0] < center_x < end_line_A[0] and last_point_y < start_line_A[1]:
                    counter_A += 1
                    points[track_id].clear()
                elif center_y > start_line_B[1] and start_line_B[0] < center_x < end_line_B[0] and last_point_y < start_line_A[1]:
                    counter_B += 1
                    points[track_id].clear()
                elif center_y > start_line_C[1] and start_line_C[0] < center_x < end_line_C[0] and last_point_y < start_line_A[1]:
                    counter_C += 1
                    points[track_id].clear()
                
        ############################################################
        ### Some post-processing to display the results          ###
        ############################################################

        
        # draw the total number of vehicles passing the lines
        cv2.putText(frame, "A", (10, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "B", (530, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "C", (910, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"{counter_A}", (270, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"{counter_B}", (620, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"{counter_C}", (1040, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frame2show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        with empty_container.container():
            c1, c2 = st.columns([0.7, 0.3])
            with c1:
                st.image(frame2show, use_column_width=True)
            with c2:
                # Create a DataFrame from the dictionary
                df = pd.DataFrame(list(trackid_dictionary.items()), columns=['Vehicle Type', 'Count'])

                # Calculate the total length of unique values for each key
                df['Count'] = df['Count'].apply(lambda x: len(set(x)))
                st.write("### Vehicle Count: ")
                st.dataframe(df, hide_index=True, width=400)

        writer.write(frame)

    # release the video capture, video writer, and close all windows
    video_cap.release()
    writer.release()
    # cv2.destroyAllWindows()

    convert_avi_to_mp4(output_video_path, final_video_path)
    st.success("Completed.")

    # display the final video
    st.video(final_video_path)
    

def main():

    c1, c2 = st.columns([0.15, 0.85], gap='small'
    with c2:
        st.title('🚕 Vehicle Counting')
        st.write('Vehicle Detection and Counting through Analytics.')
        st.divider()

    if os.path.exists(save_path):
        os.remove(save_path)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    if os.path.exists(final_video_path):
        os.remove(final_video_path)

    uploaded_file = st.file_uploader("Upload a video", type=['.mp4', '.avi'])
    if st.button('submit', key='btn-1') and uploaded_file:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
                        
        with st.spinner("Processing..."):
            video_process()

        st.divider()

if __name__ == '__main__':
    main()
