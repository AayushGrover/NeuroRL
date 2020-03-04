import cv2

path = '/media/aayush/TOSHIBA EXT/Vikas-HDD-Sem8-PE/Glioma files/2014-02-25 111155_Kanniya_optochiasmatic glioma/'
file = 'Segment01.wmv'

vidcap = cv2.VideoCapture(path+file)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("./Kanniya/Segment01-segmentation/image"+str(count).zfill(4)+".jpg", image)     # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 0.25 # it will capture image in each 0.25 second
count = 1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)