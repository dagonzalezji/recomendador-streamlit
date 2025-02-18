import gdown
import os

# IDs de archivos en Google Drive (REEMPLAZAR con tus IDs)
drive_files = {
    "models/modelo_nn_simple.h5": "1iZP89hG9OuY3ZrqTd0-sDo1xOuoH-LB6",
    "models/modelo_nn_profundo.h5": "1p9vHhcBYbyPA336HRnfV31nJrP3hAiI7",
    "models/modelo_autoencoder.h5": "1SSaeEl1j9PTnRDD1a4iqehG5VSBwg4Qf",
    "data/data_processed.csv": "1g4NpU1DjJzkMFf_E7GHeDp02SoZvXa0u"
}

def download_files():
    """ Descarga archivos de Google Drive si no existen localmente. """
    for local_path, file_id in drive_files.items():
        if not os.path.exists(local_path):
            print(f"Descargando {local_path} desde Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)
        else:
            print(f"{local_path} ya existe, omitiendo descarga.")

download_files()  # Llamar a la función al importar este archivo
