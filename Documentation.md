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
* In videos previously we used .get(____________) to get various parameters of video. Now we can use .set(______) to set that parameters value.

* setMouseCallback(image window name, function which we define for any mouse event)

*
