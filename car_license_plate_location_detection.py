# References:                                                              #
# https://gist.github.com/nikgens/1a129d620978a4abc6a9a30f5f66e0d3         #
# https://www.youtube.com/watch?v=cKD-OW3NlKA&t=195s                       #
                                                                           #
############################################################################

# kütüphaneler
import cv2
import numpy as np
import imutils

# Arayüz için gerekli fonksiyon
def nothing(x):
    pass

# Ekrana bastırılan görüntü için ölçek ayarı
scale = 2

# Orjinal görüntüyü okuma
img = cv2.imread('4.jpg')

# BGR -> HSV format dönüşümü
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

############################ Arayüz tasarımı  ######################################
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 118, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 130, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("Filter-d", "Trackbars", 4, 50, nothing)
cv2.createTrackbar("Filter-sigma_color", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Filter-sigma_space", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("Canny_Threshold_1", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Canny_Threshold_2", "Trackbars", 150, 255, nothing)

cv2.createTrackbar("Epsilon", "Trackbars", 2, 15, nothing)
cv2.createTrackbar("length(approx)", "Trackbars", 4, 8, nothing)

cv2.createTrackbar("Kernel_size", "Trackbars", 6, 50, nothing)
####################################################################################

# Sürekli While Döngüsü
while True:

    # Orjinal görüntü okuma
    img = cv2.imread('4.jpg')

    ######################### Arayüzden değer okuma ######################
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")

    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    filter_d = cv2.getTrackbarPos("Filter-d", "Trackbars")
    filter_sigma_color = cv2.getTrackbarPos("Filter-sigma_color", "Trackbars")
    filter_sigma_space = cv2.getTrackbarPos("Filter-sigma_color", "Trackbars")

    canny_threshold_1 = cv2.getTrackbarPos("Canny_Threshold_1", "Trackbars")
    canny_threshold_2 = cv2.getTrackbarPos("Canny_Threshold_2", "Trackbars")

    epsilon = cv2.getTrackbarPos("Epsilon", "Trackbars")
    length = cv2.getTrackbarPos("length(approx)", "Trackbars")

    kernel_size = cv2.getTrackbarPos("Kernel_size", "Trackbars")
    ########################################################################################



    # HSV format üzerinden beyaz renk için maske oluşturma (thresholding işlemi)
    lower_white = np.array([l_h,l_s,l_v]) #0,0,130
    upper_white = np.array([u_h,u_s,u_v]) # 179,130,255
    mask = cv2.inRange(hsv,lower_white,upper_white)

    # Oluşturulan maskeye closing morphologic operatör işlemi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thresholded_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Orjinal görüntünün revize edilmesi
    result = cv2.bitwise_and(img,img,mask=thresholded_close)

    # Revize edilen görüntünün BGR -> GRAY format dönüşümü
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Edgeleri koruyan smoothing (BilateralFilter) işlemi
    bfilter = cv2.bilateralFilter(gray, filter_d, filter_sigma_color, filter_sigma_space)

    # Canny yöntemiyle edge bulma işlemi
    edged = cv2.Canny(bfilter, canny_threshold_1, canny_threshold_2)

    #### Çıktıları ölçekleyerek ekrana bastırma #######
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    img_ = cv2.resize(img, (new_width, new_height))
    cv2.imshow("img", img_)

    new_width = int(mask.shape[1] * scale)
    new_height = int(mask.shape[0] * scale)
    mask_ = cv2.resize(mask, (new_width, new_height))
    cv2.imshow("mask", mask_)

    new_width = int(thresholded_close.shape[1] * scale)
    new_height = int(thresholded_close.shape[0] * scale)
    thresholded_close_ = cv2.resize(thresholded_close, (new_width, new_height))
    cv2.imshow("thresholded_close", thresholded_close_)

    new_width = int(result.shape[1] * scale)
    new_height = int(result.shape[0] * scale)
    result_ = cv2.resize(result, (new_width, new_height))
    cv2.imshow("result", result_)

    new_width = int(gray.shape[1] * scale)
    new_height = int(gray.shape[0] * scale)
    gray_ = cv2.resize(gray, (new_width, new_height))
    cv2.imshow("gray", gray_)

    new_width = int(bfilter.shape[1] * scale)
    new_height = int(bfilter.shape[0] * scale)
    bfilter_ = cv2.resize(bfilter, (new_width, new_height))
    cv2.imshow("filter", bfilter_)

    new_width = int(edged.shape[1] * scale)
    new_height = int(edged.shape[0] * scale)
    edged_ = cv2.resize(edged, (new_width, new_height))
    cv2.imshow("edge", edged_)
    #################################################################



    # Contours bulma işlemi
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Contourlardan dörtgen bulma işlemi
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == length:
            location = approx
            # break
            mask = np.zeros(gray.shape, np.uint8)
            x_, y_, w_, h_ = cv2.boundingRect(contour)

            # Contourun bulunduğu alanı beyazlatma işlemi
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)

            # Orjinal görüntüde sadece plakayı gösteren işlem
            new_image_2 = cv2.bitwise_and(img, img, mask=mask)

            # Plakayı ayrı bir görüntü olarak veren işlem
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            cv2.imshow('cropped_image', cropped_image)

            # Orjinal Görüntüde plakayı işaretleyen işlem
            final = img
            final[x1:x2 + 1, y1:y2 + 1] = [255,255,255]
            final = cv2.rectangle(final, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)

            #### Çıktıları ölçekleyerek ekrana bastırma #######
            new_width = int(new_image.shape[1] * scale)
            new_height = int(new_image.shape[0] * scale)
            new_image_ = cv2.resize(new_image, (new_width, new_height))
            cv2.imshow('newimage', new_image_)

            new_width = int(new_image_2.shape[1] * scale)
            new_height = int(new_image_2.shape[0] * scale)
            new_image__ = cv2.resize(new_image_2, (new_width, new_height))
            cv2.imshow('newimage_2', new_image__)

            new_width = int(final.shape[1]*scale)
            new_height = int(final.shape[0]*scale)
            final_ = cv2.resize(final,(new_width,new_height))
            cv2.imshow('final', final_)
            ####################################################

    key = cv2.waitKey(1)
    if key==27:
        break
