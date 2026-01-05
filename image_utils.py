from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # הופך את התמונה למערך (array) ומחזיר אותו
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    # 1. הפיכה לשחור-לבן (ממוצע של שלושת ערוצי הצבע)
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image

    # 2. בניית פילטרים לשינויים באנכי ובאופקי
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])

    # 3. הפעלת קונבולוציה (convolve2d) עם הדירושת של המרצה
    edgeX = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # 4. חישוב עוצמת הקצוות לפי הנוסחה שביקשו
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
