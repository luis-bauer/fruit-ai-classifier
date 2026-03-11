import time
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'models', 'customdata', 'quick.tflite')
label_path = os.path.join(BASE_DIR, 'models', 'customdata', 'label.txt')

file = open(label_path, 'r')
labels = file.readlines()

# Load TFLite model and allocate tensors.
print('Loading model...')
interpreter = tflite.Interpreter(model_path)
print('Allocating tensors....')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_batch_shape = input_details[0]['shape']
input_shape = (input_batch_shape[1], input_batch_shape[2])

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

def main():
    print(input_shape)
    while True:
        success, img = vid.read()
        if not success:
            print('Fehler, irgendetwas stimmt nicht mit der Kamera')
        else:
            img = cv2.flip(img, 1)
            img = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
        
            input_batch = np.expand_dims(img, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_batch)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])

            label_index = np.argmax(output_data)
            label = labels[label_index]
            score = round(output_data[0][label_index], 2)
            result_text = label + ' (' + str(score) + ')'

            img = cv2.resize(img, dsize=(700, 700), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.putText(img, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('bild', img)

            print(label)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        print('This is running!')
        main()
    except KeyboardInterrupt:
        print('')
        print(f'Exiting..')
