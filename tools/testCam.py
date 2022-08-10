import cv2

print("start search")
num = 0
count = 0
for i in range(0,10000):
    count=count+1
    cap = cv2.VideoCapture(i)
    if(cap.isOpened()):
        print("cam device num is %d" %i)
        num = 0
        break
    if(count%200==1):
        print("have find %d" %i)
        count = count-200
    if(i==9999):
        print("no find")
    
cap = cv2.VideoCapture(num)
while True:

    _, frame = cap.read()
    cv2.imshow("test",frame)
    cv2.waitKey(10)
    

