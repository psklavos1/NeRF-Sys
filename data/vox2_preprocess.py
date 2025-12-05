import os
import sys
import argparse
import cv2
import numpy as np
import pickle as pkl
import logging
from pathlib import Path
from einops import rearrange
import torch
import torchvision
import torchvision.transforms as T

torchvision.set_video_backend("pyav")
from torchvision.io import read_video


VOXCELEB_PATH = "data/vox2_mp4"


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    return logger


logg = get_logger(__name__)

# def extract_frames(video):
#     cap = cv2.VideoCapture(video)

#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     frames = np.empty((n_frames, h, w, 3), np.dtype('uint8'))

#     fn, ret = 0, True
#     while fn < n_frames and ret:
#         ret, img = cap.read()
#         frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         fn += 1

#     cap.release()
#     return frames


def extract_frames(video_path):
    vid_frames, _, metadata = read_video(video_path, pts_unit="sec")
    vid_frames = rearrange(vid_frames, "t h w c -> t c h w")
    return vid_frames


def save_video(path, video_id, frame):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{video_id}.vid"
    pkl.dump(frame, open(os.path.join(path, filename), "wb"))
    logg.info(f"Saved file: {filename}")


def process_video_folder():
    dataset_path = Path(VOXCELEB_PATH)
    train_vids_path = dataset_path / "dev" / "mp4"
    test_vids_path = dataset_path / "test" / "mp4"

    train_video_list, test_video_list = [], []
    for root, dirs, files in os.walk(train_vids_path):
        vid_files = [file for file in files if file.endswith(".vid")]
        if len(files) > 0 and len(vid_files) == 0:
            train_video_list.append((root, files))
            logg.info(f"Found {len(files)} files in {root}")
    logg.info(f"Found {len(train_video_list)} train videos")

    for root, dirs, files in os.walk(test_vids_path):
        vid_files = [file for file in files if file.endswith(".vid")]
        if len(files) > 0 and len(vid_files) == 0:
            test_video_list.append((root, files))
            logg.info(f"Found {len(files)} files in {root}")
    logg.info(f"Found {len(test_video_list)} test videos")

    for folder, files in train_video_list:
        norm_folder = os.path.normpath(folder)
        frames = np.concatenate(
            [extract_frames(os.path.join(norm_folder, file)) for file in files]
        )
        frames = torch.from_numpy(frames).type(dtype=torch.float)
        frames = rearrange(frames, "n h w c -> n c h w")

        save_video(os.path.dirname(norm_folder), os.path.basename(norm_folder), frames)
    logg.info(f"Finished processing train videos")

    for folder, files in test_video_list:
        norm_folder = os.path.normpath(folder)
        frames = np.concatenate(
            [extract_frames(os.path.join(norm_folder, file)) for file in files]
        )
        frames = torch.from_numpy(frames).type(dtype=torch.float)
        frames = rearrange(frames, "n h w c -> n c h w")

        save_video(os.path.dirname(norm_folder), os.path.basename(norm_folder), frames)
    logg.info(f"Finished processing test videos")


def extractImages():
    import shutil

    dataset_path = Path(VOXCELEB_PATH)
    train_vids_path = dataset_path / "dev" / "mp4"
    test_vids_path = dataset_path / "test" / "mp4"
    train_video_list, test_video_list = [], []
    for root, dirs, files in os.walk(train_vids_path):
        jpg_files = [file for file in files if file.endswith(".jpg")]
        if len(files) > 0 and len(jpg_files) == 0:
            train_video_list.append((root, files))
            logg.info(f"Found {len(files)} files in {root}")

            pathIn = os.path.join(root, files[0])
            pathOut = root
            count = 0
            vidcap = cv2.VideoCapture(pathIn)
            success, image = vidcap.read()
            success = True
            while success:
                try:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 250))  # added this line
                    success, image = vidcap.read()
                    cv2.imwrite(
                        pathOut + "/frame%d.jpg" % count, image
                    )  # save frame as JPEG file
                    count = count + 1
                    if count == 16:
                        logg.info(f"preprocessing video {root} finished")
                        break
                except:
                    logg.info(f"preprocessing video {root} failed")
                    shutil.rmtree(root)
                    break
    logg.info(f"Found {len(train_video_list)} train videos")

    for root, dirs, files in os.walk(test_vids_path):
        jpg_files = [file for file in files if file.endswith(".jpg")]
        if len(files) > 0 and len(jpg_files) == 0:
            test_video_list.append((root, files))
            logg.info(f"Found {len(files)} files in {root}")

            pathIn = os.path.join(root, files[0])
            pathOut = root
            count = 0
            vidcap = cv2.VideoCapture(pathIn)
            success, image = vidcap.read()
            success = True
            while success:
                try:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 250))
                    success, image = vidcap.read()
                    cv2.imwrite(pathOut + "/frame%d.jpg" % count, image)
                    count = count + 1
                    if count == 16:
                        logg.info(f"preprocessing video {root} finished")
                        break
                except:
                    logg.info(f"preprocessing video {root} failed")
                    shutil.rmtree(root)
                    break
    logg.info(f"Found {len(test_video_list)} test videos")


if __name__ == "__main__":
    extractImages()
