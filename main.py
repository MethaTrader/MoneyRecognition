from PIL import Image
import torch
import numpy as np
import cv2 as cv
from mss import mss

mon = {'left': 100, 'top': 100, 'width': 640, 'height': 640}
model = torch.hub.load('ultralytics/yolov5', 'custom', 'model/best.pt', force_reload=True)
balances = [1000, 500, 200, 100, 50, 20]

with mss() as sct:
    while True:
        balance = 0
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB',
            (screenShot.width, screenShot.height),
            screenShot.rgb,
        )

        out = np.array(img)

        results = model(img, size=640)  # includes NMS
        detection = results.pandas().xyxy[0].to_numpy()

        out = cv.cvtColor(out, cv.COLOR_BGR2RGB)

        out[0 : 100, 0 : 640] = (0, 0, 0)

        if detection.size > 0:
            for i in range(len(detection)):
                print(detection[i][6])
                if detection[i][4] > 0.5:
                    balance += balances[detection[i][5]]
                    cv.rectangle(out, (int(detection[i][0]), int(detection[i][1])), (int(detection[i][2]), int(detection[i][3])), (0,255,0), 2)
                    cv.putText(out, detection[i][6], (int(detection[i][0]), int(detection[i][1])-16), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2, cv.LINE_AA)
        cv.putText(out, 'Balance: ' +  str(balance) + ' UAH',(175,55), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 3, cv.LINE_AA)

        out = cv.resize(out, None, fx=1, fy=1)
        cv.imshow('test', out)

        if cv.waitKey(33) & 0xFF in (
            ord('q'),
            27,
        ):
            break
