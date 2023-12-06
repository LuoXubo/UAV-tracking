"""
@Description :   目标跟踪: 1.检测目标 2.跟踪目标 3.目标丢失后重新检测
@Author      :   Xubo Luo 
@Time        :   2023/12/04 11:58:18
"""
import cv2
import argparse
import time 

counter = 0
start_time = time.time()
font_color = (0, 0, 255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Object Tracking using OpenCV \n\n \
                                     Usage: python tracker_init.py --video_path <video_path> --tracker <tracker_id>')
    parser.add_argument('--video_path', type=str, default='pen_1.mp4', help='Path to video file or camera id')
    parser.add_argument('--tracker', type=int, default='0', help='Tracker type (0: KCF, 1: MOSSE, 2: CSRT)')

    args = parser.parse_args()

    # 选择一个视频文件或者使用摄像头
    if args.video_path == '0':
        args.video_path = 0
    cap = cv2.VideoCapture(args.video_path)

    # 选择目标初始框（Bounding Box），手动指定或者使用目标检测算法来获取
    ret, frame = cap.read()
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)

    # 选择跟踪器
    switcher = {
        0: cv2.TrackerKCF_create(),
        1: cv2.TrackerMOSSE_create(),
        2: cv2.TrackerCSRT_create(),
    }
    # switcher = {
    #     0: cv2.legacy.TrackerKCF_create(),
    #     1: cv2.legacy.TrackerMOSSE_create(),
    #     2: cv2.legacy.TrackerCSRT_create(),
    # }
    tracker = switcher.get(args.tracker, "Invalid tracker")

    # 初始化跟踪器
    tracker.init(frame, bbox)


    while True:
        # 读取当前帧
        ret, frame = cap.read()
        counter += 1
        if (time.time() - start_time) != 0:
            # cv2.putText(frame, "FPS " + str(counter / (time.time() - start_time)), 
            #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "FPS " + str(counter / (time.time() - start_time)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color)
            start_time = time.time()
            counter = 0

        if not ret:
            break

        # 更新跟踪器
        ret, bbox = tracker.update(frame)
        cv2.putText(frame, "Press r to reset.", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color)
        cv2.putText(frame, "Press q to quit.", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color)


        # 如果目标被成功跟踪
        if ret:
            # 画出目标框
            p1 = (int(bbox[0]), int(bbox[1])) # 左上
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) # 右下
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            # print('Center position: ' + str((int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))))
        else:
            # 如果跟踪失败，可以在这里添加相应的处理逻辑
            cv2.putText(frame, "Tracking lost!", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            pass

        # 显示结果
        cv2.imshow("Tracking", frame)

        # 按'q'键退出
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # 按'r'键重新初始化跟踪器
        elif key == ord('r'):
            ret, frame = cap.read()
            frame_disp = frame.copy()
            cv2.putText(frame, 'Select target ROI and press ENTER', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, font_color)
            cv2.imshow("Resetting", frame)
            bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, bbox)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
