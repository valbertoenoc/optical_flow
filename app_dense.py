import cv2
import numpy as np

from optical_flow.optical_flow import OpticalFlow

of = OpticalFlow()

def main(): 
    cap = cv2.VideoCapture('videos\\cars.mp4')
    if not cap.isOpened():
        print('[ERROR] Invalid video.')
        return

    # Take first frame and find features in it using ShiTomasi params for detection
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(old_frame)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame_count = 0
    while(1):
        ret, frame = cap.read()

        # Restart video stream when it reaches the end
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # calculate sparse optical flow - Farneback method (dense)
        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
        # display flow grid with flow direction
        of.display_flow_grid(frame_gray, flow)

        # display flow color coded by direction
        of.display_color_coded_directions(hsv, flow)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame_gray
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()