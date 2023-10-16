# from PIL import Image;
import numpy as np;
# import pandas as pd;
import cv2;
import pytesseract;
from pypdf import PdfReader, PdfWriter
import pdf2image;
import re;
import os;

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
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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

# rotating image right, not used currently but could explore if want to account for DN
def rotate_right(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# rotating image left, not used currently but could explore if want to account for DN
def rotate_left(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

# pass array of images (which should be numpy arrays) for processing: grayscale > gaussian blur > thresholding, and returns processed images
def process_images(image_arr):
    processed_images = [];
    for i, page in enumerate(image_arr):
        # load normal Image as np array
        page_arr = np.array(page)
        page_gs = get_grayscale(page_arr)
        page_gaussian = remove_noise_gaussian(page_gs)
        page_threshold = thresholding(page_gaussian)
        processed_images.append(page_threshold)
    return processed_images

# pass array of images and create results based on different image processing sequences, write to script folder as png files
def test_images(image_arr):
    for i, page in enumerate(image_arr):
        # load normal Image as np array
        page_arr = np.array(page);

        # convert normal to grayscale, write to file
        page_gs = get_grayscale(page_arr);
        status_gs = cv2.imwrite("gs-page-"+ str(i) +".png", page_gs);

        # deskew grayscale, write to file
        page_deskew = deskew(page_gs);
        status_deskew = cv2.imwrite("deskew-page-"+ str(i) +".png", page_deskew);

        # convert normal to denoise, write to file 
        # page_noise = remove_noise(page_arr);
        # status_noise = cv2.imwrite("noise-page-"+ str(i) +".png", page_noise);

        # convert normal to gaussian denoise, write to file
        # page_gaussian = remove_noise_gaussian(page_arr);
        # status_gaussian = cv2.imwrite("gaussian-page-"+ str(i) +".png", page_gaussian);

        # # convert grayscale to denoise, write to file
        # page_gs_noise = remove_noise(page_gs);
        # status_gs_noise = cv2.imwrite("gs-to-denoise-page-"+ str(i) +".png", page_gs_noise);

        # # convert grayscale to gaussian, write to file
        page_gs_gaussian = remove_noise_gaussian(page_gs);
        status_gs_gaussian = cv2.imwrite("gs-to-gaussian-page-"+ str(i) +".png", page_gs_gaussian);

        # # convert grayscale to threshold, write to file
        # page_gs_thresh = thresholding(page_gs);
        # status_gs_thresh = cv2.imwrite("gs-to-thresh-page-"+ str(i) +".png", page_gs_thresh);

        # # convert grayscale to denoise to threshold, write to file
        # page_gs_noise_thresh = thresholding(page_gs_noise);
        # status_gs_noise_thresh = cv2.imwrite("gs-to-denoise-to-thresh-page-"+ str(i) +".png", page_gs_noise_thresh);

        # convert grayscale to gaussian to threshold, write to file
        page_gs_gaussian_thresh = thresholding(page_gs_gaussian);
        status_gs_gaussian_thresh = cv2.imwrite("gs-to-gaussian-to-thresh-page-"+ str(i) +".png", page_gs_gaussian_thresh);

        # img = cv2.imread(page);
        # status = cv2.imwrite("do-page-"+ i +".png", img);

# takes in numpy image array, processes image and performs ocr on it to find id. if cant find from processed, perform ocr on original image array
def ocr_find_id(image_arr):
    id_name = ''
    processed_images = process_images(image_arr)
    for i, page in enumerate(processed_images):
        print(f'Getting data from page {i+1:d}')
        data = pytesseract.image_to_string(page)
        print(f'----------------------data------------------\n{data:s}')
        if "DELIVERY ORDER" in data.upper():
            # regex to find delivery order with "FME-DO-nnnn-xxxxxxxx" where n is number, x is any alphanumeric character
            do = re.findall(r'FME-DO-\d{4}-[A-Za-z0-9]{8}', data)
            if len(do) != 0:
                id_name = do[0]
                break
            else:
                id_name = "NOT_FOUND_DO_ID_"
        elif "CASH SALES TAX INVOICE" in data.upper():
            # regex to find cash sales with "FME-CS-nnnn-xxxxxxxx" where n is number, x is any alphanumeric character
            cs = re.findall(r'FME-CS-\d{4}-[A-Za-z0-9]{8}', data)
            if len(cs) != 0:
                id_name = cs[0]
                break
            else:
                id_name = "NOT_FOUND_CS_ID_"
        elif "PREPARED BY" in data.upper() and "RECEIVED BY" in data.upper():
            id_name = "DELIVERY_NOTE_"
            break
        else:
            id_name = "NOT_FOUND_"
    # if cannot find the id, try to perform ocr on the unprocessed image instead
    # loop is basically copy of whatever is above, just passing in different image_arr
    if "NOT_FOUND_" in id_name:
        print("ID not found from processed images, performing OCR on original image")
        for i, page in enumerate(image_arr):
            print(f'Getting data from page {i+1:d}')
            data = pytesseract.image_to_string(page)
            print(f'----------------------data------------------\n{data:s}')
            if "DELIVERY ORDER" in data.upper():
                do = re.findall(r'FME-DO-\d{4}-[A-Za-z0-9]{8}', data)
                if len(do) != 0:
                    id_name = do[0]
                    break
                else:
                    id_name = "NOT_FOUND_DO_ID_"
            elif "CASH SALES TAX INVOICE" in data.upper():
                cs = re.findall(r'FME-CS-\d{4}-[A-Za-z0-9]{8}', data)
                if len(cs) != 0:
                    id_name = cs[0]
                    break
                else:
                    id_name = "NOT_FOUND_CS_ID_"
            elif "PREPARED BY" in data.upper() and "RECEIVED BY" in data.upper():
                id_name = "DELIVERY_NOTE_"
                break
            else:
                id_name = "NOT_FOUND_"
    return id_name




# test_imgs = pdf2image.convert_from_path("FME-DO-4-COMBINED.pdf", poppler_path=r'C:\Program Files\poppler-23.08.0\Library\bin');
# test_images(test_imgs);



### MAIN CODE
# setup, hardcoded filepath
# scanned_file_arr = []
# filepath_str = 'C:\\Users\\FMEUser15\\Documents\\Repos\\fme-pdf-ocr'
# filepath = os.fsencode(filepath_str)
#
# for subdir, dirs, files in os.walk(filepath):
#     subdir_str = os.fsdecode(subdir)
#     for file in files:
#         filename = os.fsdecode(file)
#         if filename.endswith('.pdf'):
#             full_filepath = os.path.join(subdir_str, filename)
#             # print(full_filepath);
#             # scanned_file_arr.append(full_filepath);
#             img = pdf2image.convert_from_path(full_filepath, poppler_path=r'C:\Program Files\poppler-23.08.0\Library\bin')
#             print(f'----------------------image length------------------\n{len(img):d}')
#             # print(img)
#             file_id = ocr_find_id(img)
#             if "NOT_FOUND" in file_id.upper() or "DELIVERY_NOTE" in file_id.upper():
#                 file_id = file_id + filename
#             else:
#                 file_id = file_id + ".pdf"
#             print(f'----------------------file id------------------\n{file_id:s}')
#             # pdf = PdfReader(filename)
#             new_filepath = subdir_str + "\\" + file_id
#             os.rename(full_filepath, new_filepath)

### simple navigation through specified encoded filepath to find pdfs
# for file in os.listdir(filepath):
#     filename = os.fsdecode(file);
#     if filename.endswith(".pdf"):
#         print(os.path.join(filepath_str, filename));

### chunk to test image processing
# img = pdf2image.convert_from_path('C:\\Users\\FMEUser15\\Documents\\Repos\\fme-pdf-ocr\\pdfs\\NOT_FOUND_09102023102055-0012.pdf', poppler_path=r'C:\Program Files\poppler-23.08.0\Library\bin')
# test_images(img)
# file_id = ocr_find_id(img)
# print(file_id)

### splitting pdf based on end of page, only contains partial code because some of it was repurposed for the main function
# pdf = PdfReader("FME-DO-11-COMBINED.pdf");
# count = 1;
# output = PdfWriter();
# for i in range(len(pdf.pages)):
#     output.add_page(pdf.pages[i]);
#     if str(i) in end_pages:
#         print("end page found");
#         with open("%s.pdf" % end_pages[str(i)], "wb") as outputStream:
#                 output.write(outputStream);
#                 output = PdfWriter();
#         count += 1;
