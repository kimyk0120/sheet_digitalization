import numpy as np
import cv2
import PIL

def cv2imgvis(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(27)  # esc
    cv2.destroyAllWindows()


if __name__ == '__main__':

    print(cv2.version.opencv_version)

    img = cv2.imread("test_image/test_outer_detection/name_card_0.jpeg")
    ori_img = img.copy()

    h, w, c = img.shape

    r = 800.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 800)

    resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    height, width, channel = resize_img.shape

    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Edge detection
    edged = cv2.Canny(gray_img, 75, 200)

    # Find Countour
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True,)[:5]  # 외곽의 면적이 큰 순서대로 정렬

    for c in cnts:
        pass





    print("prcs fin")
