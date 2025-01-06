import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"C:\Users\Arthur\3_ambiente_de_testes\deteccao_objetos\runs\detect\train6\weights\best.pt")

image_path = r'C:\Users\Arthur\3_ambiente_de_testes\deteccao_objetos\imagens_teste\payload(1).jpg'
image = cv2.imread(image_path)

results = model.predict(r'C:\Users\Arthur\3_ambiente_de_testes\deteccao_objetos\imagens_teste\payload(1).jpg')

for result in results:
    boxes = result.boxes.xyxy.numpy()  # Coordenadas das caixas (x1, y1, x2, y2)
    classes = result.boxes.cls.numpy()  # Índices das classes

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        x_meio = int((x1 + x2) / 2)
        y_meio = int((y1 + y2) / 2)

        # Plotar o ponto na imagem
        cv2.circle(image, (x_meio, y_meio), radius=5, color=(0, 255, 0), thickness=-1)

        # Opcional: Mostrar a classe perto do ponto
        class_name = model.names[int(cls)]
        cv2.putText(image, class_name, (x_meio + 5, y_meio - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Salvar ou exibir a imagem com os pontos plotados
output_path = r'C:\Users\Arthur\3_ambiente_de_testes\deteccao_objetos\imagens_teste\resultado.jpg'
cv2.imwrite(output_path, image)
cv2.imshow("Resultado", image)
cv2.waitKey(0)
cv2.destroyAllWindows()