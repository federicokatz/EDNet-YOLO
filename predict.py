import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tiff
from edf import EDF

def main():
    parser = argparse.ArgumentParser(description='EDF prediction.')
    parser.add_argument('-m', '--model_config_path', type=str, default='cfg/yolov3.cfg')
    parser.add_argument('-c', '--model_ckpt_dir', type=str, default='weights/edf')
    parser.add_argument('-d', '--data_root_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, default='output')

    args = parser.parse_args()

    model = EDF(args.model_config_path, args.model_ckpt_dir, ensemble_option='affirmative', conf_threshold=0.4)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_root_dir, '*.tif')))
    if not files:
        print(f"No TIF files found in {args.data_root_dir}")
        return

    for tif_path in files:
        tif_name = os.path.splitext(os.path.basename(tif_path))[0]
        print(f'{"-"*10} {tif_name} {"-"*10}')
        output_path = os.path.join(args.output_dir, tif_name + '-affirmative-0.4.txt')
        process_tif(tif_path, model, output_path)

def process_tif(tif_path, model, output_path):
    img_stack = tiff.imread(tif_path)

    # Convertimos todo a formato compatible con YOLO
    def prepare_frame(frame):
        if frame.dtype == np.uint16:
            frame = (frame / 256).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)

        p2, p98 = np.percentile(frame, (2, 98))
        if p98 - p2 < 1e-6:  # evita dividir por cero
            frame = np.zeros_like(frame)
        else:
            frame = np.clip((frame - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        return frame

    H, W = img_stack[0].shape[:2]
    all_df = pd.DataFrame()
    fr_counter = 1

    for frame in tqdm(img_stack, desc='Processing frames'):
        img = prepare_frame(frame)
        boxes = model(img)

        N = len(boxes)
        if N == 0:
            fr_counter += 1
            continue

        boxes[:, [0, 2]] = np.round(boxes[:, [0, 2]].clip(0, W))
        boxes[:, [1, 3]] = np.round(boxes[:, [1, 3]].clip(0, H))
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]]

        ids = np.arange(1, boxes.shape[0] + 1).reshape(-1, 1)
        frs = np.ones(boxes.shape[0]).reshape(-1, 1) * fr_counter
        data = np.concatenate([frs, ids, boxes[:, :5], np.ones((N, 2)) * -1], axis=1)
        all_df = pd.concat([all_df, pd.DataFrame(data, columns=list(range(data.shape[1])))], ignore_index=True)

        fr_counter += 1

    # Si no hay detecciones, creamos un DataFrame vacío de 9 columnas
    if all_df.empty:
        all_df = pd.DataFrame(columns=list(range(9)))

    export_mot_df(all_df, output_path)
    print(f'Exported to {output_path}')

def export_mot_df(df, out_path):
    all_df = df.astype({
        0: 'int', 1: 'int', 2: 'int', 3: 'int', 4: 'int',
        5: 'int', 6: 'float', 7: 'int', 8: 'int'
    })
    all_df.to_csv(out_path, index=False, header=False)

if __name__ == '__main__':
    main()
