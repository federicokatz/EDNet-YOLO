import os
import glob
import itertools
import numpy as np
import tifffile as tiff
import cv2
from tqdm import tqdm
from edf import EDF


#   Normalización real para YOLO
def normalize_16bit_to_8bit(img):
    """Normalización correcta para imágenes de microscopía uint16."""
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)

    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


#   Procesamiento de Ground Truth
def binarize_and_count(gt_path):
    """Carga GT, binariza todo >0 como célula y cuenta objetos."""
    img = tiff.imread(gt_path)

    if img.ndim == 3:
        img = img[0]  # GT a veces viene como stack

    # Convertir a rango visible sin alterar labels
    if img.dtype == np.uint16:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Binarización simple
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # Conteo
    num_labels, _ = cv2.connectedComponents(binary)
    return num_labels - 1


#   PREDICCIÓN CON YOLO
def process_image_with_yolo(image_path, model):
    img_stack = tiff.imread(image_path)
    img = img_stack[0] if img_stack.ndim == 3 else img_stack

    # Normalizar uint16 a uint8
    if img.dtype == np.uint16:
        img = normalize_16bit_to_8bit(img)

    # Convertir a 3 canales para YOLO
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    boxes = model(img)
    return len(boxes)


def main():
    data_dir = './Fluo-N2DL-HeLa (1)/Fluo-N2DL-HeLa/02'
    gt_dir = './Fluo-N2DL-HeLa (1)/Fluo-N2DL-HeLa/02_GT/TRA'

    ensemble_options = ['consensus', 'unanimous', 'affirmative']
    thresholds = [0.3, 0.5, 0.7]
    param_combinations = list(itertools.product(ensemble_options, thresholds))

    image_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.tif')))

    results = {param: [] for param in param_combinations}

    print(f"\nSe procesarán {len(image_files)} imágenes")
    print(f"Total combinaciones: {len(param_combinations)}\n")

    for ensemble_option, conf_thres in param_combinations:
        print(f"\n========== Procesando ensemble='{ensemble_option}', "
              f"conf_thres={conf_thres} ==========")

        model = EDF(
            'cfg/yolov3.cfg',
            'weights/edf',
            ensemble_option=ensemble_option,
            conf_threshold=conf_thres
        )

        for i, (img_path, gt_path) in enumerate(zip(image_files, gt_files), 1):

            detected_count = process_image_with_yolo(img_path, model)
            real_count = binarize_and_count(gt_path)

            error = abs(detected_count - real_count)
            results[(ensemble_option, conf_thres)].append(error)

            print(f"[{i}/{len(image_files)}] {os.path.basename(img_path)} → "
                  f"GT={real_count} | YOLO={detected_count} | error={error}")

        avg_error = np.mean(results[(ensemble_option, conf_thres)])
        print(f"➡ Promedio de error: {avg_error:.2f}")

    print("\n================ RESUMEN FINAL ================\n")
    for param, errors in results.items():
        avg_error = np.mean(errors)
        print(f"{param}:  error_promedio = {avg_error:.2f}")

if __name__ == '__main__':
    main()
