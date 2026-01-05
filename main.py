import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

# --- פונקציות העזר (מה שביקשו ב-image_utils) ---

def load_image(path):
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    # הפיכה לאפור
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image

    # פילטרים לזיהוי קצוות
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])

    # קונבולוציה
    edgeX = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # עוצמת קצוות
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG

# --- ההרצה עצמה ---

# כאן את מקשרת את התמונה! ודאי שהשם בגרשיים זהה לשם הקובץ שהעלית בצד
my_file_name = "my_image.png" 

# 1. טעינה
original_img = load_image(my_file_name)

# 2. ניקוי רעשים
clean_img = median(original_img, ball(3))

# 3. זיהוי קצוות
edges = edge_detection(clean_img)

# 4. הפיכה לשחור לבן (בינארי)
binary_edges = edges > (np.mean(edges) * 1.5)

# 5. הצגת התוצאה
plt.imshow(binary_edges, cmap='gray')
plt.show()

# 6. שמירת התמונה למחשב (כדי שתוכלי להעלות אותה לגיטהאב)
final_result = Image.fromarray((binary_edges * 255).astype(np.uint8))
final_result.save("edge_result.png")
