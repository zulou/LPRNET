import os
import shutil

# Rutas de las carpetas
crops_dir = 'crops'
dataset_train_dir = 'preprocesed'
annotation_file = 'annotations.txt'

# Asegúrate de que la carpeta dataset/train exista
os.makedirs(dataset_train_dir, exist_ok=True)

# Abre el archivo de anotaciones
with open(annotation_file, 'r') as file:
    for line in file:
        # Divide la línea en el nombre del archivo y la etiqueta
        parts = line.strip().split()
        if len(parts) == 2:
            file_path, label = parts
            
            # Extrae el nombre del archivo sin la ruta
            file_name = os.path.basename(file_path)
            
            # Extrae el número único del nombre del archivo
            unique_number = file_name.split('_')[-1].split('.')[0]
            
            # Crea el nuevo nombre del archivo
            new_file_name = f"{label}_{unique_number}.jpg"
            
            # Ruta completa del archivo original
            old_file_path = os.path.join(crops_dir, file_name)
            
            # Ruta completa del nuevo archivo
            new_file_path = os.path.join(dataset_train_dir, new_file_name)
            
            # Mueve y renombra el archivo
            if os.path.exists(old_file_path):
                shutil.move(old_file_path, new_file_path)
                print(f"Renombrado y movido: {old_file_path} -> {new_file_path}")
            else:
                print(f"Archivo no encontrado: {old_file_path}")

print("Proceso completado.")