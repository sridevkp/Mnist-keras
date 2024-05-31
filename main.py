
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import cv2 as cv
import numpy as np

model = load_model("model.hdf5")
labels = list(range(10))

w, h = 28, 28
sw, sh = 512, 512
img = np.zeros(( sw, sh, 3 ), np.uint8 )

def handle_draw_over():
    global img
    gimg = cv.cvtColor( img, cv.COLOR_RGB2GRAY )
    gimg = cv.resize( gimg, ( w, h ), interpolation = cv.INTER_AREA)
    cv.imshow("X", cv.resize( gimg, (sw,sh), interpolation=cv.INTER_AREA) )

    X = gimg.reshape(784,1) /255.0
    Y = model.predict( np.array([X]) )
    print( labels[ Y[0].argmax() ] )

def draw_brush(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.circle(img, (x,y), 20, (255,255,255), -1)
    if event == cv.EVENT_LBUTTONUP:
        handle_draw_over()

cv.namedWindow('Draw')
cv.setMouseCallback('Draw', draw_brush)

while True:
    cv.imshow("Draw", img )

    key = cv.waitKey(1) 
    if key == ord("q"):
        break    
    elif key == ord("c"):
        img = np.zeros(( sw, sh, 3 ), np.uint8 )

cv.destroyAllWindows()
