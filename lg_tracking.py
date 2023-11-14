"""
@Description :   Tracking with lightglue
@Author      :   Xubo Luo 
@Time        :   2023/11/14 22:44:43
"""

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch, transformation, coords
from lightglue import viz2d
import torch
import cv2
import numpy as np

central_coords_x = 120
central_coords_y = 90
pt_drone = np.matrix([int(central_coords_x/2), int(central_coords_y/2), 1])
ul = np.matrix([0, 0, 1])
dr = np.matrix([central_coords_x, central_coords_y, 1])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
device = torch.device("cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

img0 = cv2.imread('pic.jpg')
img0 = img0[..., ::-1]
img0 = numpy_image_to_torch(img0)

cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()
while True:
    ret, frame = cap.read()
    img1 = frame[..., ::-1]
    img1 = numpy_image_to_torch(img1)

    feats0 = extractor.extract(img0.to(device))
    feats1 = extractor.extract(img1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    if(len(m_kpts0) < 4):
        continue

    H, _ = cv2.findHomography(m_kpts0.numpy(), m_kpts1.numpy(), cv2.RANSAC, 5.0)
    pt_sate = transformation(pt_drone.T, H)
    pt_ul = transformation(ul.T, H)
    pt_dr = transformation(dr.T, H)
    x_ul, y_ul = coords(pt_ul)
    x_dr, y_dr = coords(pt_dr)

    x, y = coords(pt_sate)
    # p1 = (int(x-central_coords_x/2), int(y-central_coords_y/2))
    # p2 = (int(x+central_coords_x/2), int(y+central_coords_y/2))

    p1 = (int(x_ul), int(y_ul))
    p2 = (int(x_dr), int(y_dr))

    # print(p1, p2)
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()