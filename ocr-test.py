from PIL import Image;
import numpy as np;
import pandas as pd;
import cv2;
import pytesseract;
import pypdf;
import pdf2image;

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

def remove_noise_gaussian(image):
    return cv2.GaussianBlur(image,(5,5),0);
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def process_images(image_arr):
    processed_images = [];
    for i, page in enumerate(image_arr):
        # load normal Image as np array
        page_arr = np.array(page);
        page_gs = get_grayscale(page_arr);
        page_gaussian = remove_noise_gaussian(page_gs);
        page_threshold = thresholding(page_gaussian);
        processed_images.append(page_threshold);
    return processed_images;

def test_images(image_arr):
    for i, page in enumerate(image_arr):
        # load normal Image as np array
        page_arr = np.array(page);

        # convert normal to grayscale, write to file
        page_gs = get_grayscale(page_arr);
        status_gs = cv2.imwrite("gs-page-"+ str(i) +".png", page_gs);

        # deskew grayscale, write to file
        page_deskew = deskew(page_gs);
        status_deskew = cv2.imwrite("deskew-page-"+ str(i) +".png", page_gs);

        # # convert normal to denoise, write to file 
        # page_noise = remove_noise(page_arr);
        # status_noise = cv2.imwrite("noise-page-"+ str(i) +".png", page_noise);

        # # convert normal to gaussian denoise, write to file
        # page_gaussian = remove_noise_gaussian(page_arr);
        # status_gaussian = cv2.imwrite("gaussian-page-"+ str(i) +".png", page_gaussian);

        # convert normal to threshold, write to file
        # page_thresh = thresholding(page_arr);
        # status_thresh = cv2.imwrite("thresh-page-"+ str(i) +".png", page_thresh);

        # # convert grayscale to denoise, write to file
        # page_gs_noise = remove_noise(page_gs);
        # status_gs_noise = cv2.imwrite("gs-to-denoise-page-"+ str(i) +".png", page_gs_noise);

        # # convert grayscale to gaussian, write to file
        # page_gs_gaussian = remove_noise_gaussian(page_gs);
        # status_gs_gaussian = cv2.imwrite("gs-to-gaussian-page-"+ str(i) +".png", page_gs_gaussian);

        # # convert grayscale to threshold, write to file
        # page_gs_thresh = thresholding(page_gs);
        # status_gs_thresh = cv2.imwrite("gs-to-thresh-page-"+ str(i) +".png", page_gs_thresh);

        # # convert grayscale to denoise to threshold, write to file
        # page_gs_noise_thresh = thresholding(page_gs_noise);
        # status_gs_noise_thresh = cv2.imwrite("gs-to-denoise-to-thresh-page-"+ str(i) +".png", page_gs_noise_thresh);

        # # convert grayscale to gaussian to threshold, write to file
        # page_gs_gaussian_thresh = thresholding(page_gs_gaussian);
        # status_gs_gaussian_thresh = cv2.imwrite("gs-to-gaussian-to-thresh-page-"+ str(i) +".png", page_gs_gaussian_thresh);

        # img = cv2.imread(page);
        # status = cv2.imwrite("do-page-"+ i +".png", img);

do_imgs = pdf2image.convert_from_path("do-sample-combined.pdf", poppler_path=r'C:\Program Files\poppler-23.08.0\Library\bin');
# print(do_img);
processed_imgs = process_images(do_imgs);

end_pages = [];
for i, page in enumerate(processed_imgs):

    data = pytesseract.image_to_string(page).upper();
    print(data);
    if "RECIPIENT" in data and "CHOP" in data and "SIGNATURE" in data:
        end_pages.append(i);
    
print(end_pages);


# img = cv2.imread("testimg.png");
# print(img);

# some_arr = np.array([1, 2, 3, 4, 5]);
# for i, num in enumerate(some_arr):
#     print(i);
#     print(num);
# some_num = some_arr[2];
# some_arr[2] += 1;
# print(some_num);
# slice_arr = some_arr[:3];
# print(slice_arr);
# print(some_arr);


