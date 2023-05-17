# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the required libraries and read the original image.
### Step2:
Translate the image.
### Step3:
Scale the image.
### Step4:
Shear the image.
### Step5:
Find reflection of image.
## Program:
```python
Developed By:jagadeeshreddy561
Register Number:212222240059
i)Image Translation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("vijay.jpg") 
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 
plt.axis("off") 
plt.imshow(input_image)
plt.show()
rows, cols, dim = input_image.shape
M= np.float32([[1, 0, 100],
                [0, 1, 200],
                 [0, 0, 1]])
translated_image =cv2.warpPerspective (input_image, M, (cols, rows))
plt.imshow(translated_image)
plt.show()

ii) Image Scaling
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("vijay.jpg") 
cv2.imshow
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 
plt.axis("off") 
plt.imshow(input_image)
plt.show()
rows, cols, dim = input_image.shape
M = np. float32 ([[1.5, 0 ,0],
                 [0, 2.0 , 0],
                  [0, 0, 2]])
scaled_img=cv2.warpPerspective(input_image, M, (cols*2, rows*2))
plt.imshow(scaled_img)
plt.show()
```
iii)Image shearing
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("vijay.jpg") 
cv2.imshow
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 
plt.axis("off") 
plt.imshow(input_image)
plt.show()
rows, cols, dim = input_image.shape
M = np. float32 ([[1.5, 0 ,0],
                 [0, 2.0 , 0],
                  [0, 0, 2]])
scaled_img=cv2.warpPerspective(input_image, M, (cols*2, rows*2))
plt.imshow(scaled_img)
plt.show()
```
iv)Image Reflection
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("vijay.jpg") 
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 
plt.axis("off") 
plt.imshow(input_image)
plt.show()
rows, cols, dim = input_image.shape
M_x=np.float32([[1,0,0],
               [0,-1,rows],
               [0,0,1]])
M_y=np.float32([[-1,0,cols],
               [0,1,0],
               [0,0,1]])
reflected_img_xaxis=cv2.warpPerspective(input_image,M_x,(cols,rows))
reflected_img_yaxis=cv2.warpPerspective(input_image,M_y,(cols,rows))
plt.imshow(reflected_img_yaxis)
plt.show()

```
v)Image Rotation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
vijay_image = cv2.imread("vijay.jpg")
vijay_image = cv2.cvtColor(vijay_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(vijay_image)
plt.show()
rows,cols,dim = vijay_image.shape
angle = np.radians(30)
M = np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rotated_img = cv2.warpPerspective(vijay_image,M,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(rotated_img)
plt.show()
```
vi)Image Cropping
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
vijay_image = cv2.imread("vijay.jpg")
vijay_image = cv2.cvtColor(vijay_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(vijay_image)
plt.show()
rows,cols,dim = vijay_image.shape
cropped_img=vijay_image[20:150,60:230]
plt.axis('off')
plt.imshow(cropped_img)
plt.show()



```
## Output:
### i)Image Translation
![model](/5.1)
### ii) Image Scaling
![model](/5.2)
### iii)Image shearing
![model](/5.3)
### iv)Image Reflection
![model](/5.4)
### v)Image Rotation
![model](/5.5)
### vi)Image Cropping
![model](/5.6)
## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
