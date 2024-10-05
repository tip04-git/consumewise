import cv2
import pytesseract
import csv
import os

# Path to your images folder
images_folder = r'C:\Users\tirum\OneDrive\Desktop\devfolio\products'

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open a CSV file for writing all the results
with open('extracted_table.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Loop through all the images in the folder
    for image_name in os.listdir(images_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_name)

            # Step 1: Load the image
            image = cv2.imread(image_path)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through contours and extract text
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cropped_image = image[y:y+h, x:x+w]

                # Extract text from the cropped region
                text = pytesseract.image_to_string(cropped_image, config='--psm 6').strip()

                # Write the text as a new row in the CSV
                writer.writerow([image_name, text])

print("Data from all images saved to 'extracted_table.csv'")
