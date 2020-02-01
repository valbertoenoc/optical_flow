import cv2
import numpy as np

class OpticalFlow():
    def __init__(self):
        # params for Lucas Kanade optical flow computation
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 4,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        # params for ShiTomasi corner detection
        self.feature_params = {
            "maxCorners": 100,
            "qualityLevel": 0.5,
            "minDistance": 7,
            "blockSize": 7
        } 

    def display_flow_grid(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

        cv2.imshow('flow grid', vis)

    def display_color_coded_directions(self, hsv, flow):
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('color coded flow',rgb)

    def display_sparse_flow(self, frame, new_points, old_points):
        mask = np.zeros_like(frame)
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            x0, y0 = new
            x1, y1 = old
            mask = cv2.line(mask, (x0, y0), (x1, y1), (0,0,255), 3)
            frame = cv2.circle(frame,(x0, y0), 5, (0,0,255),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
    
    
