import cv2
import pytesseract

#step1
# Load the image
image = cv2.imread('ingredients.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

# Save the thresholded image (optional)
cv2.imwrite('preprocessed_image.jpg', thresh)
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


#step2
# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Contours Detected', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



#step3
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    
    # Apply OCR on the cropped region
    text = pytesseract.image_to_string(cropped_image, config='--psm 6')
    print(f"Detected text: {text}")

#step4
import csv

# Open CSV file for writing
with open('extracted_table.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Loop through contours and extract text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y:y+h, x:x+w]
        
        # Extract text
        text = pytesseract.image_to_string(cropped_image, config='--psm 6').strip()
        
        # Write text as a new row in the CSV
        writer.writerow([text])

print("Data saved to 'extracted_table.csv'")
