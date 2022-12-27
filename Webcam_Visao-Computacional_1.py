import cv2 #OpenCv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Abra a webcam
capture = cv2.VideoCapture(0)
# mudar resolução pra 320x240
capture.set(3, 320)
capture.set(4, 240)

# Inicialize o algoritmo de segmentação de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

# Inicializar segmentação de Self do cvzone
segmentor = SelfiSegmentation()

while True:
    # Capture a imagem da webcam
    _, frame = capture.read()

    # Imagem 1: Converta a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Imagem 2: Detecte as bordas da imagem usando a função Canny
    edges = cv2.Canny(gray, 100, 200)

    # Imagem 3: Aplique o algoritmo de segmentação de fundo
    fgmask = fgbg.apply(frame)
    # Aplique uma máscara para destacar somente os pixels do primeiro plano
    res = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Imagem 4: removendo fundo
    green = (0, 255, 0)
    imgNoBg = segmentor.removeBG(frame, green, threshold=0.8)

    #Juntando as imagens e exibindo numa janela
    imagens = cvzone.stackImages([gray,edges,res,imgNoBg], 2, 2)
    cv2.imshow("Webcam", imagens)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a webcam e feche todas as janelas
capture.release()
cv2.destroyAllWindows()