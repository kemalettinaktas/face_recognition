import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'YoklamaResimleri'
images = []       #klasörden okunan resimler
classNames = []   #sınıftaki isimler
myList = os.listdir(path)  #sınıf klasöründeki fotolar listeye
print(myList)
for std in myList:                           #sınıftaki yoklama yapılacak öğrencilerin isimleri alındı "
                                                # foto isimlendirme formatı Ad Soyad.jpg" gibi olmalı
    curImg = cv2.imread(f'{path}/{std}')      #listedeki her bir resim okunuyor (imread)
    images.append(curImg)                       #resimler okunduktan sonra images listesi oluştu
    classNames.append(os.path.splitext(std)[0]) #sınıftaki isimler classNames listesine alındı
print(classNames)

def yoklamaYap(name):
    with open('yoklama.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def findEncodings(images):  # okunan resimleri face_recognition kütüphanesinin kullanabileceği şekilde encode ettiğimiz fonskiyon
    encodeList = []  #bu fonksiyon ile resimleri tek tek okuyup encode edip, encodlu liste olarak döndürdük   normal resimler ---> encode resimler
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #rgb formata dönüştü
        encode = face_recognition.face_encodings(img)[0]   #face_recognition kütüphanesi ile encode edildi
        encodeList.append(encode)                           #listeye eklendi
    return encodeList
encodeListKnown = findEncodings(images) #sınıf listemizde(klasörde) bulunan resimler encode edildi --> yoklaması yapılacak kişiler tanıtıldı
print('Encoding Tamam...')

cap = cv2.VideoCapture(0) # kamera açıldı
while True:
    success, img = cap.read() #kameradan görüntü karesini oku
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) #görüntüyü ölçekle 1/4
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  #rgb formatına çevir

    facesCurFrame = face_recognition.face_locations(imgS) #gelen görüntü karesinden yüz konumlarını bul
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #görüntüyü encode et

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #algılanan ve encode yüzler listesi için döngü
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #sınıftaki bilien yüzler ile okunan görüntüyü kıyasla
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #yüzler arasındaki mesafeyi hesaplama(en az mesafe en çok benzerlik)
        # print(faceDis)
        matchIndex = np.argmin(faceDis) #yüz mesafesi en düşük olan elemanın konumu

        if matches[matchIndex]: #eşleşme mı
            name = classNames[matchIndex].upper() # ismi listeden bul
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #işaretlendi ve yazıldı
            yoklamaYap(name) #yoklama listesine yazıldı

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


