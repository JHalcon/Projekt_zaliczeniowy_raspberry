"""
Moduł dotyczy głównego pliku obsługującego system. Zawiera funkcje służące do wyświetlania klatek na ekranie,
funkcje analizujące obraz oraz funkcję obsługującą komunikaty email.
"""
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
from SingleMotionDetector import SingleMotionDetector

    
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
pedestrian_cascade = cv2.CascadeClassifier("/home/pi/monitoring/haarcascade_frontalface_default.xml")

kamera1_1 = None
kamera1_2 = None
kamera2_1 = None
kamera2_2 = None
zmianaKamery = None
zmianaKamery2 = None
wersjaKamery = 1
wersjaKamery2 = 1

czas = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
ile = 0

lock1 = threading.Lock()
lock2 = threading.Lock()
lock1Kamera2 = threading.Lock()
lock2Kamera2 = threading.Lock()

app = Flask(__name__)

vs = VideoStream(src=0).start()
vs2 = VideoStream(src=1).start()
time.sleep(2.0)


#Funkcja do wysylania maili
def wyslijMail(address = 'bartlomiej-pietras@wp.pl'):
    """
    Funkcja wysyła komunikat email na podany adres w sytuacji, gdy na monitoringu zostanie wykryty ruch.
    W wysłanej wiadomości znajdują się zdjęcia z obu kamer wraz z informacją o wykrytym ruchu.
 
 
    :param address: adres, na który wysyłany jest komunikat email
    :return: int - zwracany kod (1 - sukces, 0 - błąd)
    """
    global kamera1_2, kamera2_2
    
    port = 465
    sender = 'monitoringraspberry@o2.pl'
    password = 'bartekjeremiasz'
    recieve = address
    message = """\
    Właśnie wykryto ruch! Wejdź do panelu administratora i sprawdź to jak najszybiej.
    """
    context = ssl.create_default_context()

    print("Starting to send")
    with smtplib.SMTP_SSL("poczta.o2.pl", port, context=context) as server:
        server.login(sender, password)
        
        msg = MIMEMultipart()

        msg['From']=sender
        msg['To']=recieve
        msg['Subject']="Wykryto ruch!"
            
        msg.attach(MIMEText(message, 'plain'))
        
        
        cv2.imwrite("image1.jpg", kamera1_2) 
        cv2.imwrite("image2.jpg", kamera2_2) 

        part = MIMEBase('application', "octet-stream")  
        part.set_payload(open("image1.jpg", "rb").read())  
        encoders.encode_base64(part)  
        part.add_header('Content-Disposition', 'attachment; filename="image1.jpg"') 
        msg.attach(part)  
        
        part = MIMEBase('application', "octet-stream")  
        part.set_payload(open("image2.jpg", "rb").read())  
        encoders.encode_base64(part)  
        part.add_header('Content-Disposition', 'attachment; filename="image2.jpg"') 
        msg.attach(part)  
            
        try:
            server.send_message(msg)
            print("sent email!")
            del msg
            return 1
        except:   
            del msg
            print("email not sent!")
            return 0
        
    

@app.route("/")
def index():
    """
    Funkcja wystawia plik html, który będzie dostępny pod adresem ip komputera.
    :return: plik "index.html" 
    """
    return render_template("index.html")
    
@app.route('/', methods=['POST'])
def my_form_post():
    """
    Funkcja wystawia plik html, który będzie dostępny pod adresem ip komputera.
    Jest wywoływana w przypadku zastosowania metody POST. Służy też do obsługi przycisków na stronie (zmiany trybów wyświetlania kamer).
    :return: plik "respons.html" 
    """
    global zmianaKamery, zmianaKamery2, wersjaKamery, wersjaKamery2
    
    if(request.form.get('zmiana') != None):
        zmianaKamery = request.form.get('zmiana')
        wersjaKamery *= -1
        
    if(request.form.get('monitoring') != None):
        zmianaKamery2 = request.form.get('monitoring')
        wersjaKamery2 *= -1

        
    return render_template("respons.html")


@app.route('/progressILE')
def progressILE():
    """
    Funkcja obsługuje wyświetlanie liczby rozpoznanych osób.
    """
    def generate():
        global ile
        
        yield "data:" + str(ile) + "\n\n"
    return Response(generate(), mimetype= 'text/event-stream')
    
@app.route('/KameraTryb')
def KameraTryb():
    """
    Funkcja obsługuje przycisk zmiany trybu wyświetlania pierwszej kamery (pracuje wraz z funckją my_form_post).
    """
    def generate():
        global zmianaKamery
        
        yield "data:" + str(zmianaKamery) + "\n\n"
    return Response(generate(), mimetype= 'text/event-stream')

@app.route('/MonitoringTryb')
def MonitoringTryb():
    """
    Funkcja obsługuje przycisk zmiany trybu wyświetlania drugiej kamery (pracuje wraz z funckją my_form_post).
    """
    def generate():
        global zmianaKamery2
        
        yield "data:" + str(zmianaKamery2) + "\n\n"
    return Response(generate(), mimetype= 'text/event-stream')

def kamera(frameCount):
    """
    Funkcja posiada wyłączny dostęp do pierwszej kamery. 
    Co klatkę zapisuje obraz z kamery do globalnej zmiennej.
 
    :param frameCount: ilość klatek na sekunde
    """
    global vs, kamera1_1, lock1
    time_klatka = time.time()
    while True:
                
        if(((time.time() - time_klatka) * frameCount) >= 1.0):
            time_klatka = time.time()
            
            frame = vs.read()
            
            if frame is None:
                continue
            
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                                    
            with lock1:
                kamera1_1 = frame.copy()
            
          
            
def detect_face(frameCount):
    """
    Funkcja wykorzystuje globalną zmienną zawierającą oraz z kamery. 
    Analizuje ją w poszukiwaniu twarzy i przerobioną klatkę (z zaznaczonymi ramkami) zapisuje do innej zmiennej globalnej.
    Ilość rozpoznanych twarzy jest zapisywana do kolejnej zmiennej globalnej.
 
    :param frameCount: ilość klatek na sekunde
    """
    
    global vs, kamera1_1, kamera1_2, lock1, lock2, ile

    while True:

        if(kamera1_1 is None or lock1.locked()):
            continue
            
        frame = kamera1_1.copy()
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # postaci roznych rozmiarow
        pedestrians = pedestrian_cascade.detectMultiScale(gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE)
            
        # rysowanie ramek wokol rozpoznanych postaci
        for (x,y,w,h) in pedestrians:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Czlowiek', (x + 6, y - 6), font, 0.5, (0, 
            255, 0), 1)
        
                         
        
        ile = len(pedestrians)                            
        
        with lock2:
            kamera1_2 = frame.copy()

def kamera2(frameCount):
    """
    Funkcja posiada wyłączny dostęp do drugiej kamery. 
    Co klatkę zapisuje obraz z kamery do globalnej zmiennej.
 
    :param frameCount: ilość klatek na sekunde
    """
    global vs2, kamera2_1, lock1Kamera2

    while True:
   
        frame = vs2.read()
        if frame is None:
            continue
        
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        with lock1Kamera2:
            kamera2_1 = frame.copy() 


def detect_motion(frameCount):
    """
    Funkcja wykorzystuje globalną zmienną zawierającą oraz z drugiej kamery. 
    Tworzona jest instancja klasy SingleMotionDetector, za pomocą której system rozpoznaje ruch na nastepujących po sobie klatkach. 
    Klatka z zaznaczonym ruchem jest zapisywana do innej zmiennej globalnej.
 
    :param frameCount: ilość klatek na sekunde
    """
    
    global kamera2_1, kamera2_2, lock1Kamera2, lock2Kamera2

    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    while True:
   
        if(kamera2_1 is None or lock1Kamera2.locked() or wersjaKamery2 == 1):
            continue
   
        frame = kamera2_1.copy()
        #frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        if total > frameCount:

            motion = md.detect(gray)
            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)
                              
                wyslijMail()
                time.sleep(20)

        md.update(gray)
        total += 1

        with lock2Kamera2:
            kamera2_2 = frame.copy() 

def generate():
    """
    Funkcja obsługuje wyświetlanie obrazu z pierwszej kamery.
    """
    global kamera1_1, kamera1_2, lock1, lock2, wersjaKamery

    while True:
            
        if wersjaKamery == 1:
            if kamera1_1 is None or lock1.locked():
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", kamera1_1)
            
        elif wersjaKamery == -1:
            if kamera1_2 is None or lock2.locked():
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", kamera1_2)

        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def generate3():
    """
    Funkcja obsługuje wyświetlanie obrazu z drugiej kamery.
    """
    global kamera2_1, kamera2_2, lock1Kamera2, lock2Kamera2, wersjaKamery2

    while True:
    
        flag = False
        
        if wersjaKamery2 == 1:
            if kamera2_1 is None or lock1Kamera2.locked():
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", kamera2_1)
            
        elif wersjaKamery2 == -1:
            if kamera2_2 is None or lock2Kamera2.locked():
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", kamera2_2)

        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
               
@app.route("/podglad")
def podglad():
    """
    Funkcja wysyła obraz z pierwszej kamery na serwer.
    """
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

                    
@app.route("/ruch")
def ruch():
    """
    Funkcja wysyła obraz z drugiej kamery na serwer.
    """
    return Response(generate3(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=2000,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

   
    t1 = threading.Thread(target=kamera, args=(
        args["frame_count"],))
    t1.daemon = True
    t1.start()
    
    t2 = threading.Thread(target=detect_face, args=(
        args["frame_count"],))
    t2.daemon = True
    t2.start()
    
    t3 = threading.Thread(target=kamera2, args=(
        args["frame_count"],))
    t3.daemon = True
    t3.start()
    
    t4 = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t4.daemon = True
    t4.start()
    
    

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

vs.stop()