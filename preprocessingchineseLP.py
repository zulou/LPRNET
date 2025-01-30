import os
import random

def add_suffix_to_files(directory):
    for filename in os.listdir(directory):
        # Ignorar archivos que ya tienen un guion bajo seguido de un número
        if "_" in filename and filename.split("_")[-1].split(".")[0].isdigit():
            continue

        # Asegurarse de trabajar solo con archivos .jpg
        if filename.endswith(".jpg"):
            # Generar un número aleatorio de 10 dígitos
            random_number = random.randint(1000000000, 9999999999)

            # Separar nombre base y extensión
            name, extension = os.path.splitext(filename)

            # Crear el nuevo nombre
            new_filename = f"{name}_{random_number}{extension}"

            # Renombrar el archivo
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)

            print(f"Renombrado: {filename} -> {new_filename}")

# Ruta al directorio donde están los archivos
directory_path = "valid"

# Llamar a la función
add_suffix_to_files(directory_path)
