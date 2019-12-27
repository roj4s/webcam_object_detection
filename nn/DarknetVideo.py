import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ctypes import *
from models.Heatmap import Heatmap
from tqdm import tqdm


def collect_frames(cap, dps):
    # Get the fps
    fps = int(cap.get(5))

    # adjust the detections per second to be at most the fps
    if dps > fps:
        dps = fps

    # Number of detections per frame
    fps_idps = int(fps/dps)

    # Get number of frames in the video
    frame_count = int(cap.get(7))
    print(f'Frame count: {frame_count}, FPS: {fps}')

    frames = list()
    for i in tqdm(range(0, frame_count, fps_idps), desc='Collecting Frames'):

        # Set the frame we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Get frame
        ret, frame_read = cap.read()
        if not ret:
            break

        # Convert it to RGB and store it
        frames.append(cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB))

    return frames


def generate_heatmaps(frames, thresh,  width, height, acc, hm_img_name):

    hm_generator = Heatmap()

    # Each detection is the list of detections in a frame
    heatmaps = list()

    for i in tqdm(range(0, len(frames)), desc='Generating heatmap'):
        heatmaps.append(hm_generator.gen_heatmap(
            frames[i], thresh, width, height, acc))

    # Save last heatmap
    hm_generator.save_last_heatmap(hm_img_name)

    return heatmaps


def produce_video_output(cap, heatmaps, dps, hm_video_name):
    # Get the fps
    fps = int(cap.get(5))

    # adjust the detections per second to be at most the fps
    if dps > fps:
        dps = fps

    # Number of detections per frame
    fps_idps = int(fps/dps)
    frames = list()

    # First step, collect frames
    printProgressBar('Collecting frames', 0, frame_count, length=50)
    for i in range(0, int(frame_count)):

        # Get number of frames in the video
        frame_count = int(cap.get(7))

        # Width and height of the input video
        width = int(cap.get(3))
        height = int(cap.get(4))

        # Final video object
        hmap = cv2.VideoWriter(
            f"pseudo-database/heatmaps/{hm_video_name}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Reset video to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in tqdm(range(0, frame_count), desc='Generating video'):

        ret, frame_read = cap.read()
        if not ret:
            break

        # Get the current heatmap, it's the heatmap for the next fps_idps frames
        heatmap = heatmaps[int(i / fps_idps)]
        assert(frame_read.shape == heatmap.shape)

        # Add the frame and the heatmap, the latter with alpha=0.5
        new_frame = cv2.addWeighted(frame_read, 1, heatmap, 0.5, 0.0)
        hmap.write(new_frame)


def perform_detect_video(video, acc, thresh, dps):
    cap = cv2.VideoCapture(video)

    # The first step of the detection is to collect the frames where we'll
    # be perfoming the detection upon
    frames = collect_frames(cap, dps)
    assert(len(frames))

    # The second step is to generate the heatmaps for each store frame
    heatmaps = generate_heatmaps(frames,
                                 thresh,
                                 int(cap.get(3)), int(cap.get(4)),
                                 acc,
                                 video.split('.')[0])
    assert(len(heatmaps))

    # Fourth step is to produce the final video, with the heatmaps overlapped
    produce_video_output(cap, heatmaps, dps, video.split('.')[0])


def perform_detect_cam(camera_ids, acc, thresh):

    caps = []
    for camera_id in camera_ids:
        caps.append(cv2.VideoCapture(camera_id))

    # if not cap.isOpened():
    #     cap.open(0)

    hm_generator = Heatmap()

    while(True):
        # Capture frame
        for cap in caps:
            ret, frame = cap.read()

            if not ret:
                break

            heatmap = hm_generator.gen_heatmap(
                frame, thresh, int(cap.get(3)), int(cap.get(4)), acc)

            # Save last heatmap
            hm_generator.save_last_heatmap("cam")

    # When everything done, release the capture
    for cap in caps:
        cap.release()
