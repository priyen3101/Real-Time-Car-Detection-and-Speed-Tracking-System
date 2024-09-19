#!/usr/bin/env python
# coding: utf-8

import cv2
import yaml
import os
from yaml.loader import SafeLoader
import numpy as np
import time

class Yolo_Pred_With_Speed():
    def __init__(self, onnx_model, data_yaml):
        # Load model and labels
        with open(data_yaml, mode='r') as f:
            self.data_yaml = yaml.load(f, Loader=SafeLoader)
        self.lables = self.data_yaml['names']
        self.nc = self.data_yaml['nc']
        
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.prev_positions = {}  # Store previous positions (per car ID)
        self.prev_speeds = {}  # Store previous EMA speeds (per car ID)
        self.last_update_time = time.time()  # Track time for speed calculation intervals
        self.update_interval = 3  # 3-second interval for speed calculation
        self.movement_threshold = 200  # Minimum movement in pixels to consider
        self.alpha = 0.2  # Smoothing factor for EMA

    def exponential_moving_average(self, object_id, new_speed):
        if object_id in self.prev_speeds:
            # EMA formula: S_new = alpha * new_speed + (1 - alpha) * S_old
            ema_speed = self.alpha * new_speed + (1 - self.alpha) * self.prev_speeds[object_id]
        else:
            ema_speed = new_speed  # First speed measurement

        # Store the new EMA speed
        self.prev_speeds[object_id] = ema_speed
        return ema_speed

    def predictions(self, Image):
        row, col, d = Image.shape
        
        # Fit image into square image
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[:row, :col] = Image
        
        # Get prediction from square image
        YOLO_WH_INPUT = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (YOLO_WH_INPUT, YOLO_WH_INPUT), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        pred = self.yolo.forward()

        # Flatten the array
        detection = pred[0]
        
        confidances = []
        boxes = []
        classes = []

        img_w, img_h = input_image.shape[:2]
        x_factor = img_w / YOLO_WH_INPUT
        y_factor = img_h / YOLO_WH_INPUT

        for i in range(len(detection)):
            row = detection[i]
            confidance = row[4]
            if confidance > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])
                    boxes.append(box)
                    confidances.append(confidance)
                    classes.append(class_id)
        
        confidances_np = np.array(confidances).tolist()
        boxes_np = np.array(boxes).tolist()

        # Non Maximum Suppression
        index = cv2.dnn.NMSBoxes(boxes_np, confidances_np, 0.25, 0.45)
        current_time = time.time()  # Get current time for speed calculation
        
        if len(index) > 0:
            index = index.flatten()
            for ind in index:
                x, y, w, h = boxes_np[ind]
                bb_conf = int(confidances_np[ind] * 100)
                classes_id = classes[ind]
                class_name = self.lables[classes_id]
                if classes_id in [0,3,4,5,6]:
                    # Initialize speed_kmph with 0
                    speed_kmph = 0
    
                    # Speed Calculation every 3 seconds
                    object_id = ind  # Use ind as unique ID for simplicity
                    if object_id in self.prev_positions:
                        prev_x, prev_y, prev_time = self.prev_positions[object_id]
                        delta_time = current_time - prev_time
    
                        if delta_time >= self.update_interval:
                            # Calculate pixel displacement
                            delta_x = x - prev_x
                            delta_y = y - prev_y
                            delta_distance_pixels = np.sqrt(delta_x**2 + delta_y**2)
    
                            # If movement is below threshold, keep the previous speed
                            if delta_distance_pixels < self.movement_threshold:
                                speed_kmph = self.prev_speeds.get(object_id, 0)
                            else:
                                # Convert pixels to meters (adjust conversion ratio)
                                pixel_to_meter = 0.05  # Adjust as needed
                                distance_meters = delta_distance_pixels * pixel_to_meter
    
                                # Calculate speed in meters per second (m/s) and then km/h
                                speed_mps = distance_meters / delta_time
                                speed_kmph = speed_mps * 3.6
    
                                # Apply EMA to stabilize speed
                                speed_kmph = self.exponential_moving_average(object_id, speed_kmph)
    
                            # Update speed and position in the dictionary
                            self.prev_positions[object_id] = (x, y, current_time)
                        else:
                            # Keep the last known position, do not update speed
                            speed_kmph = self.prev_speeds.get(object_id, 0)
                    else:
                        # First time seeing this object, store position and current time
                        self.prev_positions[object_id] = (x, y, current_time)
                        self.prev_speeds[object_id] = 0  # No speed yet for this object
    
                    # Display speed on the image
                    speed_text = f'Speed: {speed_kmph:.2f} km/h'
                    cv2.putText(Image, speed_text, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw bounding box and class label
                text = f'{class_name}: {bb_conf}%'
                colors = self.choose_color(classes_id)
                cv2.rectangle(Image, (x, y), (x + w, y + h), colors, 2)
                cv2.rectangle(Image, (x, y - 30), (x + w, y), colors, -1)
                cv2.putText(Image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            return Image
        else:
            return Image
        
    def choose_color(self, Id):
        np.random.seed(10)
        color = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(color[Id])