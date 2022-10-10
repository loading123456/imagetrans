import string
from tokenize import String
from typing import Any, List, Tuple
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import time
import concurrent.futures
import re
from pytesseract import Output
from PIL import Image, ImageFont, ImageDraw
import googletrans 



class Box:
    def __init__(self, rect, bgColor ) -> None:
        self.bgColor = bgColor
        self.startPoint = [rect[0], rect[1]]
        self.endPoint = [rect[0] + rect[2], rect[1] + rect[3]]
        self.centerPoint = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
        self.width = rect[2]
        self.height = rect[3]

    def getRect(self, type=0) -> List:
        if type == 0:
            return self.startPoint + [self.width, self.height]
        if type == 1:
            return self.startPoint + self.endPoint


class Area(Box):
    def __init__(self,box:Box) -> None:
        super().__init__(box.getRect(type=0), box.bgColor)
        self.X_NOUN = 0.6
        self.Y_NOUN = 0.3
        self.epX = self.height * self.X_NOUN
        self.epY = self.height * self.Y_NOUN
        self.imagePath = ''

    def insertBox(self, box:Box) -> Boolean:
        if self.__isValid(box):
            self.__update(box)
            return True
        return False

    def __isValid(self, box:Box) -> Boolean:
        if(box.bgColor == self.bgColor
            and abs(self.endPoint[0] - box.startPoint[0]) <= self.epX
            and abs(self.height - box.height) <= self.epY * 2
            and (abs(self.startPoint[1] - box.startPoint[1]) <= self.epY
                    or abs(self.centerPoint[1] - box.centerPoint[1]) <= self.epY
                    or abs(self.endPoint[1] - box.endPoint[1]) <= self.epY 
                )
        ):
            if (box.height / self.height <= 0.5 
                and box.width > box.height * 3
            ):
                return False
            return True
        return False

    def __update(self, box:Box) -> None:
        if self.startPoint[0] > box.startPoint[0]:
            self.startPoint[0] = box.startPoint[0]
        if self.startPoint[1] > box.startPoint[1]:
            self.startPoint[1] = box.startPoint[1]

        if self.endPoint[1] < box.endPoint[1]:
            self.endPoint[1] = box.endPoint[1]
        if self.endPoint[0] < box.endPoint[0]:
            self.endPoint[0] = box.endPoint[0]
        
        self.centerPoint[0] = (self.startPoint[0] + self.endPoint[0])/2
        self.centerPoint[1] =  (self.startPoint[1] + self.endPoint[1])/2

        self.height = self.endPoint[1] - self.startPoint[1]
        self.width = self.endPoint[0] - self.startPoint[0]

        self.epX = self.height *  self.X_NOUN
        self.epY = self.height * self.Y_NOUN

    def saveToJpg(self):
        self.imagePath = 'cache/' + str(self.startPoint + self.endPoint) + '.jpg'
        x, y, w, h = self.getRect()
        im = None
        if self.bgColor == 0:
            im = img[y:h, x:w]
        else:
            im = getContrastImg(img[y:h, x:w])
        cv2.imwrite(self.imagePath, im)

    def getRect(self) -> List:
        x = self.startPoint[0]
        y = int(self.startPoint[1] - (self.epY/2 
                                    if(self.startPoint[1] - self.epY/2 >= 0) 
                                    else 0))
        endX = self.endPoint[0]
        endY = int(self.endPoint[1] + (self.epY/2 
                                    if(self.endPoint[1] - self.epY/2 <= img.shape[1]) 
                                    else 0))

        return [x, y, endX, endY]


class Word(Box):
    def __init__(self, rect, bgColor, text, confident ) -> None:
        box = Box(rect, bgColor)
        super().__init__(box.getRect(0), box.bgColor)
        self.text = text
        self.confident = confident


class Line(Box):
    def __init__(self, word:Word) -> None:
        super().__init__(word.getRect(0), word.bgColor)
        self.words = [word]
        self.X_NOUN = 0.7
        self.Y_NOUN = 0.3
        self.epX = self.height * self.X_NOUN
        self.epY = self.height * self.Y_NOUN
        self.amount = 1

    def insertWord(self, word:Word) -> Boolean:
        position = self.__getPos(word)
        if position != -1:
            if position == self.amount:
                self.words.append(word)
            else:
                self.words.insert(position, word)
            self.__update(word)
            return True
        return False

    def __getPos(self, word:Word) -> int:
        if abs(self.centerPoint[1] - word.centerPoint[1]) <= self.epY:
            position = 0

            for node in self.words:
                if node.centerPoint[0] < word.centerPoint[0]:
                    position += 1
                else:
                    break
            if position == 0:
                distance = self.words[0].startPoint[0] - word.endPoint[0]
                if distance <= self.epX:
                    return 0
                return -1

            if position == self.amount:
                distance = word.startPoint[0] - self.words[-1].endPoint[0]
                if distance <= self.epX :
                    return position
                return -1
            lastDistance =  word.startPoint[0] - self.words[position - 1].endPoint[0]
            nextDistance = self.words[position].startPoint[0] - word.endPoint[0]

            if (lastDistance <= self.epX 
                and nextDistance <= self.epX
            ):
                return position
        return -1
    
    def __update(self, word:Word) -> None:
        if self.startPoint[0] > word.startPoint[0]:
            self.startPoint[0] = word.startPoint[0]
        if self.startPoint[1] > word.startPoint[1]:
            self.startPoint[1] = word.startPoint[1]

        if self.endPoint[0] < word.endPoint[0]:
            self.endPoint[0] = word.endPoint[0]
        if self.endPoint[1] < word.endPoint[1]:
            self.endPoint[1] = word.endPoint[1]
        
        self.centerPoint[0] = (self.startPoint[0] + self.endPoint[0])/2
        self.centerPoint[1] = (self.startPoint[1] + self.endPoint[1])/2
        self.epX = (self.endPoint[1] - self.startPoint[1]) * self.X_NOUN
        self.epY = (self.endPoint[1] - self.startPoint[1]) * self.Y_NOUN
        self.width = self.endPoint[0] - self.startPoint[0]
        self.height = self.endPoint[1] - self.startPoint[1]
        self.amount += 1

    def getText(self) -> String:
        text = ''
        for word in self.words:
            text += ' ' + word.text 
        if text.isupper():
            text = text.lower()
        return text[1:]

    def getRect(self) -> List:
        x = self.startPoint[0]
        y = int(self.startPoint[1] - (self.epY/2 
                                    if(self.startPoint[1] - self.epY/2 >= 0) 
                                    else 0))
        endX = self.width
        endY = self.height + int(self.epY/1.5)

        return [x, y, endX, endY]


class Paragraph:
    def __init__(self, line:Line) -> None:
        self.lines = [line]
        self.avHeight = line.height
        self.X_NOUN = 0
        self.Y_NOUN = 0
        self.epX = 0
        self.epY = line.height
        self.amount = 1

    def insertLine(self, line:Line) -> Boolean:
        position = self.__getPos(line)
        if position != -1:
            self.lines.append(line)
            self.amount += 1
            return True
        return False

    def __getPos(self, line:Line) -> int:
        if (line.startPoint[1] - self.lines[-1].endPoint[1]  <=  self.epY
                and line.startPoint[1] - self.lines[-1].endPoint[1] > 0
        ):
            if ((abs(self.lines[-1].startPoint[0] - line.startPoint[0]) 
                    <= line.epX * 5)
                or (abs(line.endPoint[0] - self.lines[-1].endPoint[0]) 
                        <= line.epX * 5)
                or (abs(line.centerPoint[0] - self.lines[-1].centerPoint[0])
                        <= line.epX * 5)
            ):
                return 1
        return -1

    def __update(seld, line:Line) -> None:
        pass

    def draw(self, img):
        self.__translate()
        for i in range(self.amount):
            if (not self.textLines[i].isspace() 
                and self.textLines[i] != ''
                and self.textLines[i].replace(" ", "") != self.lines[i].getText().replace(" ", "")
            ):
                x, y, w, h = self.lines[i].getRect()
                bg = Image.new(mode="RGBA", size=(w, h), color=(235, 235, 235))
                img.paste(bg, (x, y))

                font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf', h)
                textWidth, textHeight = font.getsize(self.textLines[i])
                if textWidth < w:
                    textWidth = w
                textBox = Image.new(mode="RGBA", size=(textWidth, textHeight ), color=(0, 0, 0, 0))
                d = ImageDraw.Draw(textBox)
                d.text((0, 0), self.textLines[i], font=font, fill=(0, 0, 0))
                textBox.thumbnail((w, 1000  ), Image.ANTIALIAS)
                textBox = textBox.crop((0, 0, w, h))

                img.paste(textBox, (x, y), textBox.convert("RGBA"))


    def __translate(self) -> None:
        text = ''
        for line in self.lines:
            text += line.getText() +  ' __1 ' 

        tText = (googletrans.Translator()
                    .translate(text, dest='vi').text)

        tText = re.sub('\\s+', ' ', tText).strip()
        self.textLines = re.split('__', tText)

        for i in range(len(self.textLines)):
            if len(self.textLines[i]) > 1 and self.textLines[i][0] == '1' :
                self.textLines[i] = self.textLines[i][1:]

        
# ====================================== Main ==============================================
img = None

def translate(imagePath, savePath):
    st = time.time()

    (contours, cContours, bImg) = getRawData(imagePath)
    boxes = getBoxes(contours, cContours, bImg)
    areas = getAreas(boxes)
    saveAreasToJpg(areas)
    words = recognized(areas)
    missBoxes = getMissBoxes(words)
    missAreas = getAreas(missBoxes)
    saveAreasToJpg(missAreas)

    words += recognized(missAreas)
    lines = getLines(words)
    paragraphs = getParagraphs(lines)

    outputImg = Image.open(imagePath).convert("RGB")

    for paragraph in paragraphs:
        paragraph.draw(outputImg)
    outputImg.save(savePath)


    print("Excuse time: ", time.time() - st)




def getRawData(imagePath) -> Tuple:
    global img

    img = cv2.imread(imagePath)
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bImg = cv2.threshold(gImg, 0, 255, cv2.THRESH_OTSU 
                                            | cv2.THRESH_BINARY_INV)[1]
    cBImage = np.where(bImg == 0 , 255, 0).astype('ubyte')
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    dilation = cv2.dilate(bImg, rect_kernel, iterations = 1)
    cDilation = cv2.dilate(cBImage, rect_kernel, iterations = 1)
    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    cContours = cv2.findContours(cDilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    return (contours, cContours, bImg)


def getBoxes(contours, cContours, bImg) -> List:
    boxes = []
    rects = getBoundingRect(contours)

    for rect in rects:
        x, y, w, h = rect
        white = np.count_nonzero(bImg[y:y+h,x:x+w] == 0)
        if (float(white) / float(w * h) >= 0.1
            and (w < bImg.shape[0]
                and h < bImg.shape[1])
        ):
            boxes.append(Box(rect, 0))

    cRects = getBoundingRect(cContours)
    for cRect in cRects:
        x, y, w, h = cRect
        white = np.count_nonzero(bImg[y:y+h, x:x+w] == 0)
        if (float(white) / float((w * h)) >= 0.1
            and (w < bImg.shape[0]
                and h < bImg.shape[1])
        ):
            boxes.append(Box(cRect, 1))
    boxes = sorted(boxes, key= lambda x: (x.bgColor, x.startPoint[0]))
    return boxes

def getBoundingRect(contours) -> List:    
    result = []
    for cnt in contours:
        result.append(cv2.boundingRect(cnt))
    return result


def getAreas(boxes) -> List:
    areas = []

    for box in boxes:
        Inserted = False
        for area in areas:
            if area.insertBox(box):
                Inserted = True
                break
        if not Inserted:
            areas.append(Area(box))
    areas = sorted(areas, key=lambda x: (x.startPoint[1]))
    return areas


def getContrastColor(rbg) -> List:
    return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

def getContrastImg(img) -> List:
    contrastImg = img.copy()
    for y in range(len(contrastImg)):
        for x in range(len(contrastImg[y])):
            contrastImg[y][x] = getContrastColor(contrastImg[y][x])
    return contrastImg    


def saveAreasToJpg(areas) -> None:
    for area in areas:
        area.saveToJpg()

def recognized(areas) -> List:
    words = []
    for area in areas:
        x, y, w, h = area.getRect()
        im = cv2.imread(area.imagePath)
        data = pytesseract.image_to_data(im, output_type=Output.DICT)
        words += toWords(data, x, y, area.bgColor)
    return words

def toWords(imgData,  sX, sY, bgColor) -> List:
    words = []
    for i in range(len(imgData['text'])):
        x, y = int(imgData['left'][i]), int(imgData['top'][i])
        w, h = int(imgData['width'][i]), int(imgData['height'][i])
        if(imgData['text'][i] != '' 
            and not imgData['text'][i].isspace()
            and w < img.shape[0]
            and h < img.shape[1]
            and imgData['conf'][i] > -1
        ): 
            text = re.sub(r"[^a-zA-Z0-9]+", ' ', 
                            imgData['text'][i])
            words.append(Word([sX + x, sY + y, w, h], 
                                bgColor,
                                imgData['text'][i],
                                imgData['conf'][i]
                                ))
    return words

def getMissBoxes(words) -> List:
    missBoxes = []
    for word in words:
        if word.confident < 50:
            missBoxes.append(Box(word.getRect(0), (word.bgColor + 1) % 2))
    return missBoxes


def getLines(words) -> list:
    lines = []
    for word in words:
        if word.confident >= 50:
            inserted = False
            for line in lines:
                if line.insertWord(word):
                    inserted = True
                    break
            if not inserted:
                lines.append(Line(word))
    lines = sorted(lines, key= lambda key: key.centerPoint[1])
    return lines 


def getParagraphs(lines) -> List:
    paragraphs = []

    for line in lines:
            inserted = False
            for paragraph in paragraphs:
                if paragraph.insertLine(line):
                    inserted = True
                    break 
            if not inserted:
                paragraphs.append(Paragraph(line))

    return paragraphs

# translate('images/1.png', 'output/1.png')
# translate('images/2.png', 'output/2.png')
# translate('images/3.png', 'output/3.png')
# translate('images/4.png', 'output/4.png')
# translate('images/5.png', 'output/5.png')
# translate('images/6.png', 'output/6.png')
# translate('images/7.png', 'output/7.png')
# translate('images/8.png', 'output/8.png')
# translate('images/9.png', 'output/9.png')
# translate('images/10.png', 'output/10.png')
# translate('images/11.png', 'output/11.png')
# translate('images/12.png', 'output/12.png')
# translate('images/13.png', 'output/13.png')
# translate('images/14.png', 'output/14.png')


