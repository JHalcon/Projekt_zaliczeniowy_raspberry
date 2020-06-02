from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, request, render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders  

#Klasa do rozpoznawania ruchu
class SingleMotionDetector:
    """
    Klasa obsługuje rozpoznawanie ruchu. Jest wykorzystywana do analizy obrazu z drugiej kamery.
    """

    def __init__(self, accumWeight=0.5):
        self.accumWeight = accumWeight
        self.bg = None

    def update(self, image):
        """
        Funkcja, docelowo wykonywana co klatkę, aktualizuje obiekt i umożliwia rozpoznawanie ruchu z wykorzystaniem nowej klatki.
     
        :param image: aktualna klatka
        """
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
            
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        """
        Funkcja analizuje ostatnie klatki w celu rozpoznania ruchu. 
     
        :param image: aktualna klatka
        :return: (thresh, (minX, minY, maxX, maxY)) - współrzędne punktów ograniczających prostokąt, w którym rozpoznano ruch
        """
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        if len(cnts) == 0:
            return None

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        return (thresh, (minX, minY, maxX, maxY))

    