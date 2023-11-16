"""
@Description :   Tracking with lightglue
@Author      :   Xubo Luo 
@Time        :   2023/11/14 22:44:43
"""

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, numpy_image_to_torch, transformation, coords, match_pair
from lightglue import viz2d
import torch
import cv2
import numpy as np

central_coords_x = 120
central_coords_y = 90
pt_drone = np.matrix([int(central_coords_x/2), int(central_coords_y/2), 1])
ul = np.matrix([0, 0, 1])
dr = np.matrix([central_coords_x, central_coords_y, 1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
# device = torch.device("cpu")

extractor = SuperPoint(max_num_keypoints=2048,
                       nms_radius=3).eval().cuda()  # load the extractor
match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval().cuda()

# img0 = cv2.imread('pic.jpg')
# img0 = img0[..., ::-1]
# img0 = numpy_image_to_torch(img0)
# feats0 = extractor.extract(img0.to(device))

tensorA, scaleA = load_image('pic.jpg', grayscale=False)

cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()
cnt = 0
while True:
    cnt = cnt + 1
    if cnt%2 != 1:
        continue
    ret, frame = cap.read()
    img1 = frame[..., ::-1]
    # img1 = numpy_image_to_torch(img1)
    tensorB = numpy_image_to_torch(img1)

    
    # feats1 = extractor.extract(img1.to(device))
    # matches01 = matcher({"image0": feats0, "image1": feats1})
    # feats0, feats1, matches01 = [
    #     rbd(x) for x in [feats0, feats1, matches01]
    # ]  # remove batch dimension

    # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    # if(len(m_kpts0) < 4):
    #     continue

    # H, _ = cv2.findHomography(m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy(), cv2.RANSAC, 5.0)

    pred = match_pair(extractor, matcher, tensorA, tensorB)
    kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    if(len(m_kpts0) < 4):
        continue
    H, _ = cv2.findHomography(m_kpts0.numpy(), m_kpts1.numpy(), cv2.RANSAC, 5.0)
    if H is None:
        continue
    # pt_sate = transformation(pt_drone.T, H)
    pt_ul = transformation(ul.T, H)
    pt_dr = transformation(dr.T, H)
    x_ul, y_ul = coords(pt_ul)
    x_dr, y_dr = coords(pt_dr)

    # x, y = coords(pt_sate)
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