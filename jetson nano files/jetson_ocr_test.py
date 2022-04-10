import cv2
""" import pytesseract

im = cv2.imread("plates10.jpg")

text = pytesseract.image_to_string(im, lang = 'eng')

print(text) """

# Adding custom options
#custom_config = r'--oem 3 --psm 6'
#pytesseract.image_to_string(img, config=custom_config)



import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('plates7.jpg')
print(result)