import dropbox
import json
import os
import sys
import re
import glob
import copy
import matplotlib.pyplot as plt
from getpass import getpass
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(f"{os.getenv('REPO_DIR')}/src")
from helpers import *

sys.path.append(f"{os.getenv('MODEL_DIR')}/src")
from dataset import *
from methods import *

def lane_detector(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    white_threshold = 200
    _, white_mask = cv2.threshold(blurred, white_threshold, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(white_mask, 50, 150)
    
    height, width = edges.shape
    roi_vertices = np.array([[(width*0.1, height), (width*0.45, height*0.6),
                             (width*0.55, height*0.6), (width*0.9, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=20, maxLineGap=50)
    
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)
    
    lane_detected = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return lane_detected

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)