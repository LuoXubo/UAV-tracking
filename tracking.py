import cv2

# 选择一个视频文件或者使用摄像头
video_path = 'video.mp4'  # 或者 0，表示使用默认摄像头

# 打开视频文件或者摄像头
cap = cv2.VideoCapture(video_path)

# 选择目标初始框（Bounding Box），你可以手动指定或者使用目标检测算法来获取
ret, frame = cap.read()
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)

# 选择跟踪器（这里使用CSRT）
tracker = cv2.TrackerCSRT_create()

# 初始化跟踪器
tracker.init(frame, bbox)

while True:
    # 读取当前帧
    ret, frame = cap.read()

    if not ret:
        break

    # 更新跟踪器
    ret, bbox = tracker.update(frame)

    # 如果目标被成功跟踪
    if ret:
        # 画出目标框
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    else:
        # 如果跟踪失败，可以在这里添加相应的处理逻辑
        pass

    # 显示结果
    cv2.imshow("Tracking", frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
