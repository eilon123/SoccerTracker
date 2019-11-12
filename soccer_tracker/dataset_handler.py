from PIL import Image, ImageDraw
import numpy as np
import os
from os.path import join, dirname
from keras_retinanet.utils.image import read_image_bgr
from auto_tqdm import tqdm

from soccer_tracker.config import PROJECT_PATH
from soccer_tracker.detection_models.retinanet.retinanet import RetinaNet
from soccer_tracker import utils

class SoccerHandler:
    def __init__(self):
        self.retinanet = RetinaNet()
        self.frames_path = join(dirname(PROJECT_PATH), r'data\frames')
        self.detections_path = join(dirname(PROJECT_PATH), r'data\detections')
        self.num_frames = len(os.listdir(self.frames_path))
        self.width = 1920
        self.height = 1080
        self.field_mask = self.get_field_mask()

    def get_frame(self, i, mask=False):
        image_path = join(self.frames_path, r'{i}.bmp'.format(i=i))

        image = read_image_bgr(image_path)
        if mask:
            return image * self.field_mask[:, :, np.newaxis]
        return image

    def get_field_mask(self):
        # field_mask
        polygon = [(0, 488), (730, 236), (self.width - 1, 300), (self.width - 1, self.height - 1), (0, self.height - 1)]
        field_mask = Image.new('L', (self.width, self.height), 0)
        ImageDraw.Draw(field_mask).polygon(polygon, outline=1, fill=1)
        field_mask = np.array(field_mask)
        return field_mask

    def detect(self):
        for frame in tqdm(range(1, self.num_frames+1)):
            image = self.get_frame(frame, mask=True)
            results = self.retinanet.detect(image)
            detection_path = join(self.detections_path, r'{i}.pkl'.format(i=frame))
            utils.save_object(results, detection_path)

    def visualize(self, image, boxes, scores, labels, threshold=0.5):
        self.retinanet.visualize(image, boxes, scores, labels, threshold)

    def get_frame_detection(self, frame):
        detection_path = join(self.detections_path, r'{i}.pkl'.format(i=frame))
        results = utils.load_object(detection_path)
        return results

    def detections_generator(self):
        for frame in tqdm(range(1, self.num_frames+1)):
            if frame >= 850:
                break
            yield self.get_frame(frame), self.get_frame_detection(frame)
