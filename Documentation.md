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
Images consist of pixels whose values are stored in arrays. Grayscale images are 2D array in which the value at a position gives the intensity of the pixel at that location. Colored images have 3 channels BGR in openCV, then it is a 3D array in which each channel's 2D matrix contains the intensity of that particular color.
The various Functions involved in OpenCV are-

* cv2.imread(path)
> reading the image, usually stored in variable

* cv2.imshow(window name, variable where image is stored)
> shows the image

* cv2.imwrite(filename, image)
> for saving the image, *filename*: A string representing the file name. The filename must include image format like .jpg, .png, etc.
*image*: It is the image that is to be saved.

* cv2.split(variable where image is stored)
> splits the image into corresponding 3 channels of image

* cv2.add(img1, img2)
* cv2.addWeighted(img1, weight1, img2. weight2, gammavalue)
> the above two functions are used for adding images as name suggests

* cv2.subract(img1, img2)

* cv2.bitwise_and(img1, img2)
* cv2.bitwise_or(img1, img2)
* cv2.bitwise_xor(img1, img2)
* cv2.bitwise_not(img1)
> bitwise operations can be applied on the images.

* cv2.resize(source, (width, height))
> used for resizing of images

* cv2.erode() method
> cv2.erode() method is used to perform erosion on the image. The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object. It is normally performed on binary images.
>**Syntax:** cv2.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]])
Parameters:
**src:** It is the image which is to be eroded .
**kernel:** A structuring element used for erosion. If element = Mat(), a 3 x 3 rectangular structuring element is used. Kernel can be created using getStructuringElement.
**dst:** It is the output image of the same size and type as src.
**anchor:** It is a variable of type integer representing anchor point and it’s default value Point is (-1, -1) which means that the anchor is at the kernel center.
borderType:** It depicts what kind of border to be added. It is defined by flags like cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, etc.
**iterations:** It is number of times erosion is applied.
**borderValue:** It is border value in case of a constant border.
**Return Value:** It returns an image.
