import cv2
import numpy as np

from optical_flow.optical_flow import OpticalFlow

of = OpticalFlow()

def main(): 
    cap = cv2.VideoCapture('cars.mp4')
    if not cap.isOpened():
        print('[ERROR] Invalid video.')
        return

    # Take first frame and find features in it using ShiTomasi params for detection
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **of.feature_params)
    
    frame_count = 0
    while(1):
        ret, frame = cap.read()

        # Restart video stream when it reaches the end
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # calculate sparse optical flow - Lucas Kanade method
        # p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **of.lk_params)

        # # get only points that had their optical flow sucessefully calculated
        # new_points = p1[st==1]
        # old_points = p0[st==1]

        # # draw found points to ilustrate their direction
        # of.display_sparse_flow(frame, new_points, old_points)

        frame_count += 1
        if frame_count % 15 == 0:
            mask = np.zeros_like(frame)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **of.feature_params)
        else:
            old_gray = frame_gray.copy()
            p0 = new_points.reshape(-1,1,2)

        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()