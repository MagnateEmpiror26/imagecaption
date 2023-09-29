import streamlit as st 
from os.path import join, dirname, realpath
from glob import glob
import numpy as np
import os
import cv2,imutils


def frames():
    os.makedirs(RESIZED_FRAMES_PATH)
    def save_frame(video_path, gap):
        images_array = []
        name = video_path.split("\\")[1].split(".")[0]
        cap = cv2.VideoCapture(video_path)
        idx = 0

        while True:
            ret, frame = cap.read()
            if ret == False:
                cap.release()
                break
            if frame is None:
                break
            else:
                if idx == 0:
                    images_array.append(frame)
                    cv2.imwrite(f"video_frames/{idx}.jpeg", frame)
                else:
                    if idx % gap == 0:
                        images_array.append(frame)
                        cv2.imwrite(f"video_frames/{idx}.jpeg", frame)
            idx += 1
        return np.array(images_array)





st.write('R204452G Tungamiraishe Mukwena')
