1. Clonar repositorio
   - git clone https://github.com/federicokatz/EDNet-YOLO.git

2. Dirigirse a raiz del repo
    - cd EDNet

3. Crear y activar un entorno virtual
   - python3 -m venv venv
   - source venv/Scripts/activate

4. Actualizar pip
   - python -m pip install --upgrade pip
   - pip install -r requirements.txt

5. Descargar pesos preentrenados
   - pip install gdown
   - gdown 1ZI31NXaKWTSpq_ToLh_osO0qpjOiwvCB -O weights/weights.zip
   - unzip weights/weights.zip -d weights

6. Descargar datasets
   - Ingresar a https://mega.nz/folder/35IQEI5B#qz8fGVat6cHro9f58uczVQ
   - Descargar Fluo-N2DL-HeLa (1).zip
   - Descargar videos.zip
   - Descomprimir ambos archivos dentro de la carpeta base del repositorio /EDNet

7. Ejecutar experimentacion
   - Para el analisis de los parametros con distintos umbrales de confianza, ejecutar: pyton analizar_parametros.py
   - Para la ejecucion del script de prueba para el dataset HeLa con unos parametros especificos, correr el script de ejecucion_hela.ipynb (los parametros se definen dentro del script, pudiendo asignar "consensus", "unanimous" o "affirmative" con distitnos umbrales de confianza)

8. Ejecutar prediccion
   - python predict.py --data_root_dir ./videos --model_config_path cfg/yolov3.cfg --model_ckpt_dir weights/edf --output_dir output
   - Para visualizar las imagenes con sus respectivas boxes generadas durante la prediccion, ejecutar generar_imagenes.ipynb, asignando la ruta con las imagenes base y la ruta con el output de la ejecucion del predict