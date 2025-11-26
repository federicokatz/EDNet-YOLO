import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tiff

from edf import EDF

def main():
    parser = argparse.ArgumentParser(description='EDF prediction on TIF stacks.')
    parser.add_argument('-m', '--model_config_path', type=str,
                        default='config/yolov3.cfg')
    parser.add_argument('-c', '--model_ckpt_dir', type=str,
                        default='weights/edf')
    parser.add_argument('-d', '--data_root_dir', type=str, required=True,
                        help='Directory containing TIF files.')
    parser.add_argument('-o', '--output_dir', type=str, default='output',
                        help='Directory to store results.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar modelo
    model = EDF(args.model_config_path, args.model_ckpt_dir)

    # Buscar todos los TIF en el directorio
    tif_files = sorted(glob.glob(os.path.join(args.data_root_dir, '*.tif')))
    if not tif_files:
        print(f"No TIF files found in {args.data_root_dir}")
        return

    for tif_path in tif_files:
        print(f"\n{'Processing ' + os.path.basename(tif_path):-^80}")
        output_path = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(tif_path))[0] + '.txt'
        )
        process_tif(tif_path, model, output_path)

def process_tif(tif_path, model, output_path):
    """Procesa un TIF con múltiples frames y guarda resultados en MOT .txt"""
    img_stack = tiff.imread(tif_path)

    H, W, _ = img_stack[0].shape  # dimensiones del primer frame
    all_df = pd.DataFrame()
    fr_counter = 1  # contador de frames global

    for frame in tqdm.tqdm(img_stack, desc='Processing frames'):
        img = frame
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
        data = np.concatenate(
            [frs, ids, boxes[:, :5], np.ones((N, 2)) * -1], axis=1
        )

        all_df = pd.concat([all_df, pd.DataFrame(data)])
        fr_counter += 1

    export_mot_df(all_df, output_path)
    print(f"Exported to {output_path}")

def export_mot_df(df, out_path):
    all_df = df.astype({
        0: 'int',
        1: 'int',
        2: 'int',
        3: 'int',
        4: 'int',
        5: 'int',
        6: 'float',
        7: 'int',
        8: 'int',
    })
    all_df.to_csv(out_path, index=False, header=False)

if __name__ == '__main__':
    main()
