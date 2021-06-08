from keras.models import load_model
import cv2
import numpy as np
from os import path
import pyimgur

import pusher
pusher_client = pusher.Pusher(app_id=u'1212624', key=u'697cbb11f7b06a834a40', secret=u'eaa19fe4b69094485c76', cluster=u'ap1')


#class_names = ['Violence', 'NonViolence'] # fill the rest 
class_names = ['NonViolence', 'Violence'] # fill the rest
#class_names = ['NonViolence', 'Violence', 'Fight', 'Group', 'Hankshaking', 'hug', 'office', 'Talking'] # fill the rest
print("--------model start ---------------")


model = load_model('model.h5') 


print("------------model ends--------------")

print('---------------entereing compilation--------------')
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
            
#               metrics=['accuracy'])
print('-------------exiting compilation-----------')
#print(model.history())
#print(model.lr())
print(model.loss)
#print(model.accuracy)


img1 = None
img2 = None
img3 = None
img4 = None
img5 = None
img6 = None
img7 = None
img8 = None
img9 = None
img10 = None
img11 = None
img12 = None
img13 = None
img14 = None
img15 = None
img16 = None
img17 = None
img18 = None
img19 = None
img20 = None
img21 = None
img22 = None
img23 = None
img24 = None
img25 = None

imgNormalColor = None


imgfinal1 = None
imgfinal2 = None
imgfinal3 = None
imgfinal4 = None
imgfinal5 = None




def uploadDetection():
    CLIENT_ID = "37d2226a63a3f7c"
    PATH = "new.jpg"

    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image(PATH, title="Uploaded with PyImgur")

    print(uploaded_image.link)
    pusher_client.trigger(u'violencedetect', u'detect', uploaded_image.link)


count3 = 0
def predict():

    # imgfinal1 = np.reshape([img1,img2,img3,img4,img5], [5,112,112,3])
    # imgfinal2 = np.reshape([img6,img7,img8,img9,img10], [5,112,112,3])
    # imgfinal3 = np.reshape([img11,img12,img13,img14,img15], [5,112,112,3])
    # imgfinal4 = np.reshape([img16,img17,img18,img19,img20], [5,112,112,3])
    # imgfinal5 = np.reshape([img21,img22,img23,img24,img25], [5,112,112,3])
    imgfinal1 = np.reshape([img1,img1,img1,img1,img1], [5,112,112,3])
    imgfinal2 = np.reshape([img2,img2,img2,img2,img2], [5,112,112,3])
    imgfinal3 = np.reshape([img3,img3,img3,img3,img3], [5,112,112,3])
    imgfinal4 = np.reshape([img4,img4,img4,img4,img4], [5,112,112,3])
    imgfinal5 = np.reshape([img5,img5,img5,img5,img5], [5,112,112,3])
    # imgfinal1 = np.reshape([img1,img2,img3,img4,img5, img1,img2,img3,img4,img5], [10,112,112,3])
    #classes = np.argmax(model.predict([[imgfinal5, imgfinal4, imgfinal3, imgfinal2, imgfinal1]]), axis = -1)
    #
    # 
    classes = np.argmax(model.predict([[imgfinal1, imgfinal2, imgfinal3, imgfinal4, imgfinal5]]), axis = -1)
    print(classes)

    count=0

    for i in classes:
        if i > 0:
            count+=1

    if count > 3:
        count=0
        cv2.imwrite('new.jpg', imgNormalColor)
        return True

    names = [class_names[i] for i in classes]
    #print(names)
    return False




# def loadcompletedmodels():

def updateopticalflow(img, prev_gray):
    
    return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    img = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return img

count = 0
count2=0
counterpredict = 0
mirror=False
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("E:\Final.avi")
cam.set(3, 1280)
cam.set(4, 720)
ret_val, img = cam.read()
prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
while True:
    ret_val, img = cam.read()
    
    if mirror: 
        img = cv2.flip(img, 1)
    
    mask = np.zeros_like(img)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    imgNormalColor = img
    cv2.imshow('my webcam 2', img)
    
    if(count == 0):
        img1 = updateopticalflow(img, prev_gray)
        cv2.imshow('my webcam', img1)
        img1 = cv2.resize(img1,(112,112))
        
    if (count == 1):
        img2 = updateopticalflow(img, prev_gray)
        cv2.imshow('my webcam', img2)
        img2 = cv2.resize(img2,(112,112))
    
    if (count == 2):
        img3 = updateopticalflow(img, prev_gray)
        cv2.imshow('my webcam', img3)
        img3 = cv2.resize(img3,(112,112))
    
    if (count == 3):
        img4 = updateopticalflow(img, prev_gray)
        cv2.imshow('my webcam', img4)
        img4 = cv2.resize(img4,(112,112))

    if (count == 4):
        img5 = updateopticalflow(img, prev_gray)
        cv2.imshow('my webcam', img5)
        img5 = cv2.resize(img5,(112,112))
        count = 0
        if (predict()):
            counterpredict+=1
        else:
            counterpredict=0
    if (counterpredict >3):
        #upload image to imgur
        print('detection')
        #uploadDetection()
        counterpredict=0


    count += 1
   

    if cv2.waitKey(1) == 27: 
        break  # esc to quit
cv2.destroyAllWindows()

#imgfinal = np.reshape([img1], [-1,112,112,3])


# imgfinal1 = np.reshape([img1,img1,img1,img1,img1, img1, img1, img1, img1, img1], [10,112,112,3])
# imgfinal2 = np.reshape([img2,img2,img2,img2,img2, img2, img2, img2, img2, img2], [10,112,112,3])
# imgfinal3 = np.reshape([img3,img3,img3,img3,img3, img3, img3, img3, img3, img3], [10,112,112,3])
# imgfinal4 = np.reshape([img4,img4,img4,img4,img4, img4, img4, img4, img4, img4], [10,112,112,3])
# imgfinal5 = np.reshape([img5,img5,img5,img5,img5, img5, img5, img5, img5, img5], [10,112,112,3])
# imgfinal6 = np.reshape([img6,img6,img6,img6,img6, img6, img6, img6, img6, img6], [10,112,112,3])
# imgfinal7 = np.reshape([img7,img7,img7,img7,img7, img7, img7, img7, img7, img7], [10,112,112,3])
# imgfinal8 = np.reshape([img8,img8,img8,img8,img8, img8, img8, img8, img8, img8], [10,112,112,3])
# imgfinal9 = np.reshape([img9,img9,img9,img9,img9, img9, img9, img9, img9, img9], [10,112,112,3])
# imgfinal10 = np.reshape([img10,img10,img10,img10,img10, img10, img10, img10, img10, img10], [10,112,112,3])

# classes = np.argmax(model.predict([[imgfinal1, imgfinal2, imgfinal3, imgfinal4, imgfinal5, imgfinal6,imgfinal7, imgfinal8, imgfinal9, imgfinal10]]), axis = -1)
# print(classes)
# names = [class_names[i] for i in classes]
# print(names)


# img1 = cv2.imread('1.jpg')
# img1 = cv2.resize(img1,(112,112))


# img2 = cv2.imread('2.jpg')
# img2 = cv2.resize(img2,(112,112))


# img3 = cv2.imread('3.jpg')
# img3 = cv2.resize(img3,(112,112))


# img4 = cv2.imread('4.jpg')
# img4 = cv2.resize(img4,(112,112))


# img5 = cv2.imread('5.jpg')
# img5 = cv2.resize(img5,(112,112))



# img6 = cv2.imread('6.jpg')
# img6 = cv2.resize(img6,(112,112))


# img7 = cv2.imread('7.jpg')
# img7 = cv2.resize(img7,(112,112))


# img8 = cv2.imread('8.jpg')
# img8 = cv2.resize(img8,(112,112))


# img9 = cv2.imread('9.jpg')
# img9 = cv2.resize(img9,(112,112))


# img10 = cv2.imread('10.jpg')
# img10 = cv2.resize(img10,(112,112))