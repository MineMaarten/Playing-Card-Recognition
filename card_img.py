"""
Card Recognition using OpenCV
Code from the blog post 
http://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python

Aangepast door Henk-Jan van Uffelen en Maarten Kok voor de cursus Beeldherkenning te Hogeschool Utrecht.
Dit script kan een gefotografeerde speelkaart herkennen. Dit gebeurt aan de hand van trainingsplaatjes (opgeslagen in de map trainingCardFolder).
Dit zijn 52 foto's van alle verschillende kaarten. Deze zijn zo genaamd dat:

Harten aas = h1
harten 2 = h2
etc.
harten boer = h11
harten vrouw = h12
harten koning = h13

De uitkomst van het script is in ditzelfde format.


Usage:
  ./card_img.py inputCardFile trainingCardFolder
Example:
  ./card_img.py /home/student/Eindopdracht/EindopdrachtGereed/inputs/input1.jpg /home/student/Eindopdracht/EindopdrachtGereed/trainingCardFolder/

"""

import sys
import numpy as np
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/") 
import cv2
import bhutils as bh

###############################################################################
# Utility code from 
# http://git.io/vGi60A
# Thanks to author of the sudoku example for the wonderful blog posts!
###############################################################################

def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

###############################################################################
# Image Matching
###############################################################################
def preprocess(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2 )
  n= 51
  kernel = np.ones((n,n), np.uint8)
  image2 = cv2.dilate(blur,kernel,iterations =1)
  image2 = cv2.erode(image2,kernel, iterations =1)
  diff =  image2 - blur
  ret,thresh = cv2.threshold(diff,30,255, cv2.THRESH_BINARY)
  return thresh
  
def imgdiff(img1,img2):
  img1 = cv2.GaussianBlur(img1,(5,5),5)
  img2 = cv2.GaussianBlur(img2,(5,5),5)
  diff = cv2.absdiff(img1,img2)
  diff = cv2.GaussianBlur(diff,(5,5),5)
  flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
  return np.sum(diff), diff

def find_closest_card(img, trainingPath):
    bestMatch = "h1"
    bestDiff = 100000000
    secondBestDiff = 100000000
    secondBestMatch = "h1"
    bestImage = img + 0
    img = preprocess(img)
    for suit in ['h','s','r','k']:
        for i in range(1, 14):
            cardName = suit + '%d' %  i
            cardImg = cv2.imread(trainingPath + cardName + ".jpg")
            if cardImg is not None:
                for card in getCards(cardImg, 1): #Als het goed is zal deze for loop maar een keer uitgevoerd worden
                    processedCard = preprocess(card)
                    turnedCard = processedCard + 0
                    for side in range(1,5):
                        newDiff, diffImage = imgdiff(turnedCard, img)
                        print("Difference of card " + cardName + " is: %d" % newDiff)
                        if newDiff < bestDiff:
                            if bestMatch != cardName:
                                secondBestDiff = bestDiff
                                secondBestMatch = bestMatch
                                secondBestImage = bestImage
                            bestImage = diffImage
                            bestDiff = newDiff
                            bestMatch = cardName
                        elif newDiff < secondBestDiff and bestMatch != cardName:
                            secondBestImage = diffImage
                            secondBestDiff = newDiff
                            secondBestMatch = cardName
                        turnedCard = bh.rotate(processedCard, side * 90)
    print(("De beste match is " + bestMatch + " met een verschil van %d. De twee na beste match had een verschil van %d. Dit was de "+ secondBestMatch) % (bestDiff, secondBestDiff))
    cv2.imshow("second best match", secondBestImage)
    cv2.imshow("best match", bestImage)
    cv2.waitKey(0)
    return bestMatch

###############################################################################
# Card Extraction
###############################################################################  
def getCards(im, numcards=4):
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 
       
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Haal de contouren van het plaatje op, bestaande uit een paar honderd punten per contour.

  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards] #Sorteer deze op grootte, en behoud slechts 1 contour ( [:numcards] ).

  for card in contours:
    peri = cv2.arcLength(card,True)
    squareContour = cv2.approxPolyDP(card,0.02*peri,True) #Benader de contour bestaande uit honderden punten met een contour die slechts 4 punten bevat, een rechthoek

    if squareContour.shape[0] != 4: #Vang een error op, en laat aan de gebruiker zien waar het fout gaat.
        print "Contour gevonden met punten ongelijk aan 4! Punten: %d" % squareContour.shape[0]
        box = np.int0(squareContour)
        cv2.drawContours(im,[box],0,(255,255,0),6)
        imx = cv2.resize(im,(1000,600))
        cv2.imshow("foute contour" ,imx)
        cv2.waitKey(0)
        continue

    approx = rectify(squareContour) #Zet de contour in een plaatje van 450x450 pixels

    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

    transform = cv2.getPerspectiveTransform(approx,h)
    warp = cv2.warpPerspective(im,transform,(450,450))

    yield warp

if __name__ == '__main__':
  if len(sys.argv) == 3:
    inputFile = sys.argv[1]
    trainingFolder = sys.argv[2]

    print(inputFile)
    im = cv2.imread(inputFile)


    cards = [find_closest_card(c, trainingFolder) for c in getCards(im,1)]

   # print cards

  else:
    print __doc__