import cv2
import numpy as np
from deepface import DeepFace
import time
import matplotlib.pyplot as plt
from collections import defaultdict

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Mülakat adaylarının duygularını ve cevaplarını saklamak için bir sözlük
candidates_data = defaultdict(lambda: {"emotions": defaultdict(list), "answers": []})

# Mülakat adayları listesi
candidates = ["Aday 1", "Aday 2", "Aday 3"]

# Mülakat süresi (saniye cinsinden)
question_duration = 15

# Mülakat soruları listesi
questions = ["xyz?", "abc?", "123?", "456?"]

# Mülakat süreci
for candidate in candidates:
    for question_index, question in enumerate(questions):
        capture = cv2.VideoCapture(0)
        start_time = time.time()

        while True:
            _, img = capture.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (130, 20, 16), 4)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                # Yüz tanıma ve duygusal analiz yap
                results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

                # Her bir yüz için duygusal analiz sonuçlarını al
                for result in results:
                    emotion = result['dominant_emotion']
                    candidates_data[candidate]["emotions"][question_index].append(emotion)  # Duyguyu ekle

                    # Sonucu ekrana yazdır
                    print(f"{candidate} adayının duygusu: {emotion}")
                    # Duyguyu kareye yazdır
                    cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                i = 0
                for (ex, ey, ew, eh) in eyes:
                    i += 1
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (80, 235, 45), 2)
                    if i == 2:
                        break

            # Soruyu ve süreyi gösterme
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = question_duration - elapsed_time

            cv2.putText(img, f"Sure: {remaining_time:.0f} saniye", (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"{candidate} icin Soru {question_index + 1}: {question}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Mulakat", img)
            k = cv2.waitKey(30) & 0xFF

            if k == 27 or elapsed_time >= question_duration:  # Çıkış veya soru süresi dolduysa döngüden çık
                break

        capture.release()
        cv2.destroyAllWindows()

        if question_index < len(questions) - 1:
            print(f"{question_index + 1}. soru bitti, devam etmek için Enter'a basın.")
            input()  # Kullanıcının Enter'a basmasını bekleyin

# Mülakat adaylarının duygu analizi sonuçlarını görselleştirme
for candidate, data in candidates_data.items():
    for question_index, emotions in data["emotions"].items():
        plt.figure(figsize=(8, 6))
        plt.hist(emotions, bins=7, color='skyblue', edgecolor='black')
        plt.title(f"{candidate} - {question_index + 1}.soru için Duygular")
        plt.xlabel('Duygular')
        plt.ylabel('Sıklık')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
