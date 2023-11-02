import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import keyboard

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Configura la resolución deseada
width = 1280
height = 720

# Inicializa la cámara con la resolución configurada
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'CH', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'LL', 14: 'M', 15: 'N', 16: 'O', 17: 'P', 18: 'Q', 19: 'R',
    20: 'S', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'
}

# Variables para almacenar la palabra actual y las letras reconocidas
current_word = ""
recognized_letters = []

# Variables para el control de velocidad
last_recognition_time = time.time()
recognition_interval = 5.0  # Tiempo mínimo entre reconocimientos (en segundos)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        # Calcula el tiempo transcurrido desde el último reconocimiento
        current_time = time.time()
        time_since_last_recognition = current_time - last_recognition_time

        # Verifica si ha pasado suficiente tiempo desde el último reconocimiento
        if time_since_last_recognition >= recognition_interval:
            # Agrega la letra reconocida a la lista y actualiza la palabra actual
            recognized_letters.append(predicted_character)
            current_word = "".join(recognized_letters)

            # Actualiza el tiempo del último reconocimiento
            last_recognition_time = current_time

        # Si se presiona la barra espaciadora, agrega un espacio
        if keyboard.is_pressed('space'):
            recognized_letters.append(' ')
            current_word = "".join(recognized_letters)

        # Si se presiona la tecla "r", reinicia la palabra
        if keyboard.is_pressed('r'):
            recognized_letters = []
            current_word = ""

        # Si se presiona la tecla "Esc", sale del programa
        if keyboard.is_pressed('esc'):
            break

        # Dibuja la letra actual en la parte superior de la pantalla y la palabra actual justo debajo
        cv2.putText(frame, f'Letra actual: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f'Palabra actual: {current_word}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
