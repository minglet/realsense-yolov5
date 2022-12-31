import cv2
# program that can be captured from a video
# video path
vidcap = cv2.VideoCapture('./test_field8.MP4')

count = 413

while(vidcap.isOpened()):
    ret, image = vidcap.read()

    # image = cv2.resize(image, (1024, 768))

    # 30 frame per a second
    if(int(vidcap.get(1)) % 20 == 0):
        print('Saved frame number :' + str(int(vidcap.get(1))))
        #saved image path
        cv2.imwrite("C:/Users/user/Gopro1_cap/image%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1

    # image size changed 4:3

vidcap.release()