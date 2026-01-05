import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
from PIL import Image

def main():
    # 1. טעינת התמונה - וודאי שהעלית קובץ בשם my_image.png לגיטהאב!
    # אם לתמונה שלך יש שם אחר, תשני את השם בגרשיים כאן למטה
    input_path = "my_image.png" 
    
    try:
        image = load_image(input_path)
    except Exception as e:
        print(f"Error: Could not find or load {input_path}")
        return

    # 2. ניקוי רעשים בעזרת פילטר חציוני (כמו שביקשו בתרגיל)
    clean_image = median(image, ball(3))

    # 3. הרצת פונקציית זיהוי הקצוות שכתבנו ב-image_utils
    edge_mag = edge_detection(clean_image)

    # 4. הפיכה לבינארי (שחור-לבן מוחלט) לפי ערך סף
    # חישוב סף אוטומטי לפי ממוצע העוצמה
    threshold = np.mean(edge_mag) * 1.5
    binary_output = edge_mag > threshold

    # 5. הצגת התמונות (כדי שתוכלי לראות שהצליח לך)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Edge Detection Result")
    plt.imshow(binary_output, cmap='gray')
    plt.axis('off')
    plt.show()

    # 6. שמירת התוצאה כקובץ PNG
    final_img = Image.fromarray((binary_output * 255).astype(np.uint8))
    final_img.save("edge_result.png")
    print("Success! 'edge_result.png' has been created.")

if __name__ == "__main__":
    main()
