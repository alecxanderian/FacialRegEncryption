# FacialRegEncryption
Uses a facial image as a base then compares it to the face currently focused using eucliean distance and FaceNet. Then gives user an option to either decrypt or encrypt a msg.

The program is quite choppy and has a major loss in fps when focused on a face, I did not optimize it.


This program was heavily outsourced from this GitHub

https://github.com/akshaybahadur21/FaceDetection

I just recoded some parts and made it simpler to fit the use case of the project.

Download Cmake:
https://cmake.org/download/

Download Visual Studio 2022 C++ DevKit:
https://visualstudio.microsoft.com/downloads/

The following python dependencies are required in order to run the program:
dlib
opencv
numpy
base64
cryptography
imutils
pickle
hashlib
