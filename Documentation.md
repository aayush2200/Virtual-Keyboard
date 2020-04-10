# Virtual Keyboard
_Aayush Shrivastava_

## Week 1 (First four Days)
Learned about Github like how to create a repo, make a branch, commit changes and merge to master branch.

How to dual boot and install ubuntu and use various linux commands.

How to edit markdown(.md) files
    
**Notes** (Markdown File)
    
* (#) for heading
* _ or * (withoutspace) for Italics
*  ** for bold
* for ordered listing simply use numbers
* use * for unordered listing
* use ~~ for text with cut
* for adding Links we can use [] where inside square brackets comes the name which will show on the link then comes () inside this comes the link. If we want to add like if someone hovers over the link then show some text then we can add text in " " that need to be displayed inside the brackets after the link with a space.
* similarly for adding images just add ! before in the link procedure.
* To write any piece of code with higlighting use three aposhtrophe signs(') then the programming language and you must end with three aphostrophe signs when you are finished with your code.
* to add any quotes use > sign.
* we can use three dash or three astreick signs to add horizontal line    
* [For more tricks](https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf)

---

## Week 1 (last three days)

>Learning python programming language

first 2 days were reading the python official [documentation](https://docs.python.org/3/tutorial/index.html) the topics included -
1. basic data types like lists strings and expressions using python IDLE.
2. Read about control flow tools.
     - if, else and elif statements and their syntax
     - for and while loops
     - defining functions and other useful functions
3. The various data structures
     - using lists as stack, queue.
     - list comprehensions.
     - tuples, sets and dictionaries
     - looping techniques
4. Reading about formatting output and reading and writing data from a file.
5. Errors and exception handling which include try, except and raise statements.
6. Object Oriented proframming
     - classes and instances 
     - inheritance
     - Private variables 
     - generators and iterators

Next we have to go through the basics, data science tutorials and advanced tutorials(first half) from this [site](https://www.learnpython.org/) which covered the bascis of python, numpy and a bit of panda.
In the last from this [site](https://scipy-lectures.org/) covered the first topic which included intro to important libraries of python like numpy, matplotlib and scipy.

The last two sites were advanced and long so it extended to mid of next week.

## Week 2

In this week Our major focus was on the Assignment which was assigned as a team of 4. The assignment consisted of four problems in which each member has to one problem.
I did the 4th problem.
Most of the time was spent in debugging as problem asked for implementing code without loops of inbuilt functions.
Also practiced some questions on [Hackerrank](https://www.hackerrank.com/domains/python).

## Week 3

In this week the first two days were kept to the solutions of the assignment and submission of other teams solution.

Then we started Image Processing with OpenCV library of python from these [video](https://www.youtube.com/watch?v=kdLM6AOd2vc&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K) tutorials and from [these](https://www.geeksforgeeks.org/opencv-python-tutorial/).

**Notes** (OpenCV)
* imread(imagename, flag)
> flag can be 1 (load colored image) or 0 (load grayscale image) or -1 (load unchanged image)
> the function returns None if wrong file name is given.

* imshow(window name, variable in which imread was stored)
> to show the image but it si shown only for a millisecond

* waitKey(millisecond to show image or if given 0 the window is not closed unless user does it so)
> capture the key we press, can do so by storing in a variable
> use mask of 0xFF i.e. AND(&) with 0xFF for e.g. k = cv2.waitKey(0) & 0xFF

* imwrite(image name which you have to save, variable where we read the image)

* VideoCapture(name of the video file or 0 if you want to use camera)
> 0 or -1 for using camera can use 1 or 2 if multiple cameras are connected
> store it in a object lets say cap ( cap = cv2.VideoCapture(0) )

* Here is the code for video bascis
``` python
import cv2

cap = cv2.VideoCapture(0)                                       #capturing the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')                        # getting four cc code
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))    # to save a video store it in variable (filename to save, fourcc code, frame rate, frame width and height in a tuple

print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()                                     # ret stores boolean if frame received true and store it in frame object
    if ret == True:
       print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                 # To get various properties of the frame
       print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

       out.write(frame)                                         # for saving each frame 

       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)           # can be used to convert color of video
       cv2.imshow('frame', gray)                                # shows video

       if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    else:
        break

cap.release()                                                   # necessary to release the captured frame in the end
out.release()
cv2.destroyAllWindows()
```
* for drawing various objects on image
```python
import numpy as np
import cv2

#img = cv2.imread('lena.jpg', 1)
img = np.zeros([512, 512, 3], np.uint8)                                 # for drwaing total black image

img = cv2.line(img, (0,0), (255,255), (147, 96, 44), 10)          # (image, start pt, end pt, color in BGR, line thickness)
img = cv2.arrowedLine(img, (0,255), (255,255), (255, 0, 0), 10)    

img = cv2.rectangle(img, (384, 0), (510, 128), (0, 0, 255), 10)     #(image, top left corner, right bottom corner, color, thickness)
img = cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)           # (image, centre, radius, color, thickness)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'OpenCv', (10, 500), font, 4, (0, 255, 255), 10, cv2.LINE_AA)    # (image, text, start pt of text, font style, font size, color, thickness, line style)
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)        

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
* In videos previously we used .get() to get various parameters of video. Now we can use .set() to set that parameters value.

* setMouseCallback(image window name, function which we define for any mouse event)

* createTrackbar(name of trackbar, window of whose trackbar we want to create, lower value, upper value, function to be called when trackbar value changes)

* getTrackbarPos(trackbar name, window in which we want to place trackbar)

* We can use bitwise operations on images also use cv2.bitwise_..... select from the options shown in pyCharm.

* Object Detection and Object Tracking Using HSV Color Space
```python
import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0);

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    #frame = cv2.imread('smarties.png')
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                # Converting BGR to HSV

    l_h = cv2.getTrackbarPos("LH", "Tracking")                  # Setting trackbars to get lower HSV values and upper HSV values
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])                             # Creating Mask
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)              # Applying mask

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

* Morphological transformations
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

dilation = cv2.dilate(mask, kernal, iterations=2)
erosion = cv2.erode(mask, kernal, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, mg, th]

for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
```

* Thresholding and Adaptive thresholding
> thresholding compares the pixel values at each point with given value and replaces that pixel according to the thresholding method done.

> Adaptive thresholding is a better way than normal thresholding it looks around a block and calculates mean or gaussian around the block and uses that as threshold value for that block.

```python
import cv2 as cv
import numpy as np

img = cv.imread('sudoku.png',0)
_, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)          # Applying normal threshold (source of image, threshold val, max val, type of threshold)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)        # Adaptive threshold (source of image, maxval, adaptive threshold method, threshold method, size of block, constant which is subracted)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

cv.imshow("Image", img)
cv.imshow("THRESH_BINARY", th1)
cv.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

cv.waitKey(0)
cv.destroyAllWindows()
```
