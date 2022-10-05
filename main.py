## Video reading code: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
## Color Thresholding: https://www.geeksforgeeks.org/detection-specific-colorblue-using-opencv-python/
## Dominant color using K-Means: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

## for improvements, I applied morphological operations and area thresholding to generate better object masks,
## bounding boxes are also drawn in red with rectangles determined from contour regions
import cv2
import numpy as np
from sklearn.cluster import KMeans

# if __name__ == '__main__':
#     x = "ECE_180_DA_DB"
#     if x == "EE_180_DA_DB":
#         print("You are living in 2017")
#     else:
#         x = x + "- Best class ever"
#         print(x)


def viewImage(img):
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def object_detection(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_low = np.array([110, 50, 50])
    blue_high = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_img, blue_low, blue_high)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel2 = np.ones((30, 30), np.uint8)
    mask = cv2.dilate(mask, kernel2, iterations=2)


    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        if cv2.contourArea(c) < 300:
            continue
        cv2.contourArea(c)
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img

if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    while vid.isOpened():
        ret, frame = vid.read()
        img = object_detection(frame)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    img = cv2.imread("180DA-WarmUp/test.jpg")
    viewImage(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=3)  # cluster number
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    viewImage(bar)




