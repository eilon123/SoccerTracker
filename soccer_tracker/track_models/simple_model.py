import cv2
import numpy as np

from soccer_tracker.track_models.sort import Sort


import sys
sys.path.append(r'C:\Users\Admin\sem 9\project\soccer_tracker')
import matplotlib.pyplot as plt
from IPython.display import display
from soccer_tracker.dataset_handler import SoccerHandler
#from soccer_tracker.track_models.simple_model import SimpleTracker
import soccer_tracker
class SimpleTracker:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.COLORS = np.random.randint(0, 255, size=(4000, 3))
        self.age = 30
        self.mot_tracker = Sort(max_age=self.age)

    def track(self):
        out = cv2.VideoWriter(f'track_age_{self.age}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (1920, 1080))

        results_generator = self.data_handler.detections_generator()
        for image, (boxes, scores, labels) in results_generator:
            boxes, scores, labels = SimpleTracker.postprocess(boxes, scores, labels)
            track_bbs_ids = self.mot_tracker.update(boxes[0])
            track_bbs_ids = track_bbs_ids.astype(np.int)

            # draw a bounding box rectangle and label on the image
            for tr in track_bbs_ids:
                color = [int(c) for c in self.COLORS[tr[-1]]]
                x1 = tr[0]
                y1 = tr[1]
                x2 = tr[2]
                y2 = tr[3]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, str(tr[-1]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(image)
            yield image

        out.release()

    @staticmethod
    def get_hw(box):
        return np.abs(box[0] - box[2]), np.abs(box[1] - box[3])

    @staticmethod
    def calculate_iou(bb1, bb2):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb1[0], bb2[0])
        yy1 = np.maximum(bb1[1], bb2[1])
        xx2 = np.minimum(bb1[2], bb2[2])
        yy2 = np.minimum(bb1[3], bb2[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        iou = wh / ((bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
                  + (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]) - wh)
        return iou

    @staticmethod
    def postprocess(boxes, scores, labels):
        boxes, scores, labels = boxes[0], scores[0], labels[0]
        boxes, scores, labels = boxes[scores != -1], scores[scores != -1], labels[scores != -1]
        relevent_indices = []

        num_boxes = len(boxes)
        for cur_index in range(num_boxes):
            w, h = SimpleTracker.get_hw(boxes[cur_index])
            right_size = 8 < w < 100 and 25 < h < 170 and 1.2 < h / w < 3.3
            if scores[cur_index] < 0.06 or not right_size:
                continue
            flag = True

            remove_indices = []
            for relevent_index in relevent_indices:
                iou = SimpleTracker.calculate_iou(boxes[cur_index], boxes[relevent_index])
                if iou >= 0.4:
                    remove_indices.append(relevent_index)
                    if labels[cur_index] != 0 and labels[relevent_index] == 0:
                        flag = False
                    elif scores[cur_index] < scores[relevent_index]:
                        flag = False

            if flag:
                for index in remove_indices:
                    relevent_indices.remove(index)
                relevent_indices.append(cur_index)

        relevent_indices = np.array(relevent_indices)

        return boxes[relevent_indices][np.newaxis], scores[relevent_indices][np.newaxis], labels[relevent_indices][
            np.newaxis]


if __name__ == '__main__':
    data_handler = SoccerHandler()
    tracker = SimpleTracker(data_handler)
    for image_tracked in tracker.track():
        pass
