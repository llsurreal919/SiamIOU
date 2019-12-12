from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import random

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from Util import Util

torch.set_num_threads(1)

class MOTracker(object):
    def __init__(self, args, sigma_l=0.4, sigma_iou=0.5, sigma_min=4, sigma_keep=30, sigma_h=0.9, back_thre=0.6, use_sot=7):
        self.sigma_l = sigma_l
        self.sigma_iou = sigma_iou
        self.sigma_min = sigma_min
        self.sigma_keep = sigma_keep
        self.sigma_h = sigma_h
        self.back_thre = back_thre
        self.use_sot = use_sot
        self.args = args
        self.util = Util()
        self.sot_tracker = self.build_sot()
        print('Tracker initialization completed...')

    def build_sot(self):
        # load config
        cfg.merge_from_file(self.args.config)
        cfg.CUDA = torch.cuda.is_available()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(self.args.snapshot,
            map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        tracker = build_tracker(model)
        return tracker

    def backward_pre(self, previous_bbox, current_bbox, current_frame, previous_frame, println=False):
        sot_back_trace = []
        frame_cur = cv2.imread(self.frame_name_list[current_frame])
        current_bbox = list(map(int, current_bbox))
        self.sot_tracker.init(frame_cur, current_bbox, back=True)
        if println:
            print('Starting sot tracker backward prediction...\tcurrent frame:{}  previous frame:{}'.format(current_frame+1, previous_frame+1))
        frame_out = cv2.imread(self.frame_name_list[previous_frame])
        backward_bbox = self.sot_tracker.track(frame_out)
        sot_back_trace.append((backward_bbox['bbox'], previous_frame+1))
        return sot_back_trace

    def forward_track(self, bbox, start, end, println=False):
        sot_trace = []
        flag = True
        frame_cur = cv2.imread(self.frame_name_list[start])
        cur_bbox = list(map(int, bbox))
        self.sot_tracker.init(frame_cur, cur_bbox)
        if println:
            print('Starting sot tracker forward prediction...\tstart frame:{}  end frame:{}'.format(start+1, end+1))

        frame_out = cv2.imread(self.frame_name_list[end])
        outputs = self.sot_tracker.track(frame_out)
        pre_bbox = list(map(float, outputs['bbox']))
        pre_score = outputs['best_score']
        sot_trace.append((pre_bbox, end+1))
        if println:
            print(sot_trace)
        return sot_trace,pre_score

    def mot_track(self, detections, frame_name_list):
        self.frame_name_list = frame_name_list
        tracks_active = []
        tracks_finished = []
        track_pend = []
        cnt = 1
        for frame_num, detections_frame in enumerate(detections, start=1):
            if frame_num ==  1:
                # create the first frame's tracklets
                dets = [det for det in detections_frame if det['score'] >= self.sigma_l]
                new_tracks = []
                for det in dets:
                    new_tracks.append({'bboxes': [(det['bbox'],frame_num)], 'max_score': det['score'], 'start_frame': frame_num, 'id':cnt, 'sot_use_time':0})
                    cnt += 1
                tracks_active = new_tracks
                continue

            if frame_num%50 == 0:
                print(frame_num)
            dets = [det for det in detections_frame if det['score'] >= self.sigma_l]
            updated_tracks = []
            for track in tracks_active:
                if len(dets) > 0:
                    best_match = max(dets, key=lambda x: self.util.iou(track['bboxes'][-1][0], x['bbox']))
                    if self.util.iou(track['bboxes'][-1][0], best_match['bbox']) >= 0.7: 
                        track['sot_use_time'] = 0
                        track['bboxes'].append((best_match['bbox'], frame_num))
                        track['max_score'] = max(track['max_score'], best_match['score'])
                        updated_tracks.append(track)
                        del dets[dets.index(best_match)]
                    else:
                        if track['sot_use_time'] < self.use_sot:
                            pre_bboxs, pre_score = self.forward_track(bbox=track['bboxes'][-1][0], start=track['bboxes'][-1][1]-1, end=frame_num-1)
                            best_match = max(dets, key=lambda x: self.util.iou(pre_bboxs[-1][0], x['bbox']))
                            if self.util.iou(best_match['bbox'], pre_bboxs[-1][0]) >= self.sigma_iou: 
                                sot_back_trace = self.backward_pre(track['bboxes'][-1][0], best_match['bbox'], frame_num-1, track['bboxes'][-1][1]-1,println=False)
                                if self.util.iou(sot_back_trace[-1][0], track['bboxes'][-1][0]) >= self.back_thre:
                                    Lambda = pre_score/(pre_score + best_match['score'])
                                    best_match['bbox'][0] = (Lambda * pre_bboxs[-1][0][0] + (1 - Lambda) * best_match['bbox'][0])
                                    best_match['bbox'][1] = (Lambda * pre_bboxs[-1][0][1] + (1 - Lambda) * best_match['bbox'][1]) 
                                    best_match['bbox'][2] = (Lambda * pre_bboxs[-1][0][2] + (1 - Lambda) * best_match['bbox'][2]) 
                                    best_match['bbox'][3] = (Lambda * pre_bboxs[-1][0][3] + (1 - Lambda) * best_match['bbox'][3])            
                                    track['bboxes'].append((best_match['bbox'], frame_num))
                                    track['max_score'] = max(track['max_score'], best_match['score'])
                                    track['sot_use_time'] = 0
                                    updated_tracks.append(track)
                                    del dets[dets.index(best_match)]
                                else:
                                    if self.util.iou(pre_bboxs[-1][0], track['bboxes'][-1][0]) >= self.sigma_iou:
                                        track['bboxes'].append((pre_bboxs[-1][0], frame_num))
                                        track['sot_use_time'] += 1
                                        updated_tracks.append(track)
                            else:
                                if self.util.iou(pre_bboxs[-1][0], track['bboxes'][-1][0]) >= self.sigma_iou:
                                    track['bboxes'].append((pre_bboxs[-1][0], frame_num))
                                    track['sot_use_time'] += 1
                                    updated_tracks.append(track)
                else:
                    if track['sot_use_time'] < self.use_sot:
                        pre_bboxs, pre_score = self.forward_track(bbox=track['bboxes'][-1][0], start=track['bboxes'][-1][1]-1, end=frame_num-1)
                        if self.util.iou(track['bboxes'][-1][0], pre_bboxs[-1][0]) >= self.sigma_iou: 
                            sot_back_trace = self.backward_pre(track['bboxes'][-1][0], pre_bboxs[-1][0], frame_num-1, track['bboxes'][-1][1]-1,println=False)
                            if self.util.iou(sot_back_trace[-1][0], track['bboxes'][-1][0]) >= self.back_thre:
                                track['bboxes'].append((pre_bboxs[-1][0], frame_num))
                                track['sot_use_time'] += 1
                                updated_tracks.append(track)
                if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                    if track['max_score'] >= self.sigma_h and len(track['bboxes']) >= self.sigma_min:
                        track_pend.append(track)

            keep = []
            for track in track_pend:
                if frame_num - track['bboxes'][-1][1] > self.sigma_keep:
                    if len(track['bboxes']) >= self.sigma_min:
                        tracks_finished.append(track)
                    else:
                        continue
                elif len(dets) == 0:
                    keep.append(track)
                else:
                    best_match = max(dets, key=lambda x: self.util.iou(track['bboxes'][-1][0], x['bbox']))
                    if self.util.iou(track['bboxes'][-1][0], best_match['bbox']) >= 0.7: 
                        track['sot_use_time'] = 0
                        track['bboxes'].append((best_match['bbox'], frame_num))
                        track['max_score'] = max(track['max_score'], best_match['score'])
                        updated_tracks.append(track)
                        del dets[dets.index(best_match)]
                    else:
                        pre_bboxs, pre_score = self.forward_track(bbox=track['bboxes'][-1][0], start=track['bboxes'][-1][1]-1, end=frame_num-1)
                        best_match = max(dets, key=lambda x: self.util.iou(pre_bboxs[-1][0], x['bbox']))
                        if self.util.iou(best_match['bbox'], pre_bboxs[-1][0]) >= self.sigma_iou: 
                            sot_back_trace = self.backward_pre(track['bboxes'][-1][0], best_match['bbox'], frame_num-1, track['bboxes'][-1][1]-1,println=False)
                            if self.util.iou(sot_back_trace[-1][0], track['bboxes'][-1][0]) >= self.back_thre:
                                Lambda = pre_score/(pre_score + best_match['score'])
                                best_match['bbox'][0] = (Lambda * pre_bboxs[-1][0][0] + (1 - Lambda) * best_match['bbox'][0])
                                best_match['bbox'][1] = (Lambda * pre_bboxs[-1][0][1] + (1 - Lambda) * best_match['bbox'][1]) 
                                best_match['bbox'][2] = (Lambda * pre_bboxs[-1][0][2] + (1 - Lambda) * best_match['bbox'][2]) 
                                best_match['bbox'][3] = (Lambda * pre_bboxs[-1][0][3] + (1 - Lambda) * best_match['bbox'][3])
                                track['bboxes'].append((best_match['bbox'], frame_num))
                                track['max_score'] = max(track['max_score'], best_match['score'])
                                track['sot_use_time'] = 0
                                updated_tracks.append(track)
                                del dets[dets.index(best_match)]
                            else:
                                keep.append(track)
                        else:
                            keep.append(track)

            track_pend = keep 
            # create new tracks
            new_tracks = []
            for det in dets:
                new_tracks.append({'bboxes': [(det['bbox'],frame_num)], 'max_score': det['score'], 'start_frame': frame_num, 'id':cnt, 'sot_use_time':0})
                cnt += 1

            tracks_active = updated_tracks + new_tracks
        tracks_finished += [track for track in tracks_active if len(track['bboxes']) >= self.sigma_keep]
        tracks_finished += [track for track in track_pend if len(track['bboxes']) >= self.sigma_keep]
        tracks_trimmed = self.interp_tracks(tracks_finished)  
        return tracks_trimmed

    def interp_tracks(self, tracks_finished):
        furnished_tracks = []
        
        for ftracks in tracks_finished:
            tem = []
            starting_frame = ftracks['bboxes'][0][1]
            ending_frame = ftracks['bboxes'][-1][1]
            interp_track = np.zeros((ending_frame - starting_frame + 1, 4))
            
            frames_present = []

            for fframe in ftracks['bboxes']:
                interp_track[fframe[1] - starting_frame, :] = fframe[0]
                frames_present.append(fframe[1])
            
            frames_present_abs = (np.array(frames_present) - starting_frame).tolist()
            frames_missing = [f for f in range(starting_frame, ending_frame + 1) if f not in frames_present]
            frames_missing_abs = (np.array(frames_missing) - starting_frame).tolist()
            for i in range(4):
                interp_track[frames_missing_abs, i] = np.interp(frames_missing, frames_present, interp_track[frames_present_abs, i])
            for f in range(ending_frame - starting_frame +1):
                tem.append(interp_track[f, :].tolist())
            furnished_tracks.append({'bboxes': tem, 'max_score': ftracks['max_score'], 'start_frame': starting_frame})
        
        return furnished_tracks

    def uadetrac_wrapper(self, tracks):
        out = []
        for track in tracks:
            tem = []
            for i, bbox in enumerate(track['bboxes']):
                tem.append(bbox[0])
            out.append({'bboxes': tem, 'max_score': track['max_score'], 'start_frame': track['start_frame']})
        return out

