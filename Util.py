import numpy as np
import csv
import os

class Util(object):

    def load_mot(self, detections):
        data = []
        if type(detections) is str:
            raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
        else:
            assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
            raw = detections.astype(np.float32)

        end_frame = int(np.max(raw[:, 0]))
        for i in range(1, end_frame+1):
            idx = raw[:, 0] == i
            bbox = raw[idx, 2:6]
            scores = raw[idx, 6]
            dets = []
            num = 0
            for bb, s in zip(bbox, scores):
                dets.append({'bbox': [bb[0], bb[1], bb[2], bb[3]], 'score': s, 'num': num })
                num += 1
            data.append(dets)
        return data 

    def save_to_csv(self, out_path, tracks):
        with open(out_path, "w") as ofile:
            field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']

            odict = csv.DictWriter(ofile, field_names)
            id_ = 1
            for track in tracks:
                for i, bbox in enumerate(track['bboxes']):
                    row = {'frame': track['start_frame'] + i,
                        'id': id_,                       
                        'x': bbox[0],
                        'y': bbox[1],
                        'w': bbox[2],
                        'h': bbox[3],
                        'score': track['max_score'],
                        'wx': -1,
                        'wy': -1,
                        'wz': -1}

                    odict.writerow(row)
                id_ += 1

    def save_to_ua(self, out_path, tracks):
        with open(out_path, "w") as ofile:
            field_names = ['x', 'y', 'w', 'h', 'frame', 'id']
            odict = csv.DictWriter(ofile, field_names)
            id_ = 1
            for track in tracks:
                for i, bbox in enumerate(track['bboxes']):
                    row = {                 
                        'x': bbox[0],
                        'y': bbox[1],
                        'w': bbox[2],
                        'h': bbox[3],
                        'frame': track['start_frame'] + i, 
                        'id': id_
                        }

                    odict.writerow(row)
                id_ += 1

    def iou(self, boxA, boxB):
        assert len(boxA) == len(boxB), 'the len of boxes are not same {}, {}'.format(len(boxA), len(boxB))
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        if xA < xB and yA < yB:
            interArea = (xB - xA) * (yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0
        return iou

    def init_video(self, filedir, filename):
        video_folder = os.path.join(filedir, filename)
        frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
        frame_name_list = [os.path.join(video_folder, '') + s for s in frame_name_list]
        frame_name_list.sort()
        return frame_name_list