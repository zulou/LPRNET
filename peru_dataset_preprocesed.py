import cv2
import os
from pathlib import Path

# Configuración de rutas (cambiar según necesidad)
source_dir = Path("dataset_peru_raw")
target_dir = Path("dataset_peru_preprocesed")

# Crear carpeta de destino si no existe
target_dir.mkdir(parents=True, exist_ok=True)

# Extensiones de imagen soportadas
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Procesar todas las imágenes en la carpeta de origen
for img_file in source_dir.iterdir():
    try:
        # Verificar si es un archivo de imagen
        if img_file.suffix.lower() in valid_extensions:
            # Leer imagen
            img = cv2.imread(str(img_file))
            
            if img is not None:
                # Redimensionar imagen
                resized_img = cv2.resize(img, (94, 24), interpolation=cv2.INTER_LINEAR)
                
                # Guardar imagen redimensionada
                output_path = target_dir / img_file.name
                cv2.imwrite(str(output_path), resized_img)
                print(f"Imagen procesada: {img_file.name}")
            else:
                print(f"Error al leer: {img_file.name}")
        else:
            print(f"Archivo ignorado (no es imagen): {img_file.name}")
    except Exception as e:
        print(f"Error procesando {img_file.name}: {str(e)}")

print("Proceso completado!")