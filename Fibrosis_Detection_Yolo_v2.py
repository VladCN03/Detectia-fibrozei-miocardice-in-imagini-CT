import os
import cv2
import numpy as np
import pydicom
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# Setări HU pentru detecția fibrozei
HU_MIN = 60
HU_MAX = 90

# Modelul YOLO pre-antrenat pentru detectarea fibrozei
model_path = 'C:/Users/Vlad Capraru/Descarcarile mele/Sem II Anul 3/Proiect IRA2/RezultateYOLO/fibrosis_zone_detector3/weights/best.pt'
model = YOLO(model_path)

# Foldere input/output
dicom_folder = 'C:/Users/Vlad Capraru/Descarcarile mele/Sem II Anul 3/Proiect IRA2/All_CT_Images'
mask_folder = 'C:/Users/Vlad Capraru/Descarcarile mele/Sem II Anul 3/Proiect IRA2/All_CT_Masks'
save_folder = 'C:/Users/Vlad Capraru/Descarcarile mele/Sem II Anul 3/Proiect IRA2/Fibrosis_Output_Images'
temp_folder = 'C:/Users/Vlad Capraru/Descarcarile mele/Sem II Anul 3/Proiect IRA2/Temp_YOLO_Input'
os.makedirs(save_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# Salvare rezultate într-un fișier Excel
results_summary = []
summary_file = os.path.join(save_folder, 'fibrosis_summary.xlsx')

# Citirea imaginilor DICOM și conversia în HU
def read_dicom_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image_array = dicom_data.pixel_array.astype(np.float32)
    slope = getattr(dicom_data, 'RescaleSlope', 1)
    intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    image_array = image_array * slope + intercept
    return image_array, dicom_data

# Funcția pentru detectarea fibrozei pe baza HU
def detect_fibrosis_by_hu(subregion, hu_min=60, hu_max=90):
    mask = np.where((subregion >= hu_min) & (subregion <= hu_max), 255, 0).astype(np.uint8)
    return mask

# Procesarea imaginilor DICOM cu YOLO și algoritmul HU
for file in os.listdir(dicom_folder):
    if not file.lower().endswith('.dcm'):
        continue

    dicom_path = os.path.join(dicom_folder, file)
    image_array, dicom_data = read_dicom_image(dicom_path)

    # Creăm o versiune PNG pentru YOLO
    temp_img = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    temp_png_path = os.path.join(temp_folder, file.replace('.dcm', '.png'))
    cv2.imwrite(temp_png_path, temp_img)

    # Aplicăm YOLO pentru detectarea inițială
    results = model.predict(source=temp_png_path, conf=0.5, save=False, imgsz=640)
    fibrosis_detected = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        subregion = image_array[y1:y2, x1:x2]

        # Aplicăm detecția de fibroză bazată pe HU
        fibrosis_mask = detect_fibrosis_by_hu(subregion, HU_MIN, HU_MAX)

        if np.sum(fibrosis_mask) > 0:
            fibrosis_detected = True
            # Calculăm procentul de fibroză
            fibrosis_pixels = np.sum(fibrosis_mask > 0)
            total_pixels = fibrosis_mask.size
            fibrosis_percent = (fibrosis_pixels / total_pixels) * 100

            # Salvăm rezultatul în listă pentru Excel
            results_summary.append((file, f"{fibrosis_percent:.2f}".replace('.', ',')))

            # Creăm o imagine color pentru overlay
            overlay_image = cv2.cvtColor(temp_img[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(fibrosis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 1)

            # Salvăm imaginea rezultată
            save_path = os.path.join(save_folder, file.replace('.dcm', '_fibrosis.png'))
            cv2.imwrite(save_path, overlay_image)
            print(f'Salvat imaginea cu fibroză: {save_path}')
            break

    # Salvăm și imaginile fără fibroză
    if not fibrosis_detected:
        no_fibrosis_path = os.path.join(save_folder, file.replace('.dcm', '_no_fibrosis.png'))
        cv2.imwrite(no_fibrosis_path, temp_img)
        results_summary.append((file, '0,00'))
        print(f'⚠️ {file}: Fibroză nu a fost detectată. Salvare imagine fără fibroză: {no_fibrosis_path}')

# Salvare rezultate în Excel
df = pd.DataFrame(results_summary, columns=['filename', 'fibrosis_percent'])
df.to_excel(summary_file, index=False)
print(f'✅ Rezumatul a fost salvat la: {summary_file}')

print('Detectie finalizata.')
