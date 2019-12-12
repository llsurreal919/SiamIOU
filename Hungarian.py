class Hungarian():
    def __init__(self,vis,link,tracks,dets):
        self.vis = vis
        self.link = link
        self.tracks = tracks
        self.dets = dets
        self.M = []
        
    def match(self):
        res = 0
        for track in self.tracks:
            res += self.find(track)
        return res,self.M
    
    def find(self,track):
        best_match = sorted(self.dets, key=lambda x:iou(track['bboxes'][-1], x['bbox']), reverse=True)
        for y in best_match:
            if self.vis[y['num']] == 0 and iou(track['bboxes'][-1], y['bbox'])>=sigma_iou:
                self.vis[y['num']] = 1
                if self.link[y['num']] == 0:
                    self.link[y['num']] = track
                    self.M.append((track, y))
                    return 1
                else:
                    self.M.remove(self.link[y['num']], y)
                    if self.find(self.link[y['num']]):
                        self.link[y['num']] = track
                        self.M.append((track, y))
                        return 1
        return 0