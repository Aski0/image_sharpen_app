# Importowanie wymaganych bibliotek
from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
import os
import io
import uuid
import logging
from tensorflow.keras.models import load_model

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# 🔧 Konfiguracja ścieżek do modeli AI
MODEL_PATHS = {
    "Model 1": "model1000x512_b1_ep30.h5",
    "Model 2": "model800x2k_b1_ep10.h5",
    "Model 3": "model1000x512_b1_ep10.h5",
    "Model 4": "model4.h5",
    "Model 5": "model5.h5",
}

# Bufory do przechowywania załadowanych modeli i dostępnych nazw
MODELS = {}
AVAILABLE_MODEL_NAMES = []

# Bufor cache przetworzonych obrazów (klucz: session_id)
PROCESSED_IMAGES_CACHE = {}

#  Funkcja do ładowania modeli przy starcie aplikacji
def load_application_models():
    app.logger.info("Rozpoczynam ładowanie modeli...")
    for name, path in MODEL_PATHS.items():
        try:
            if os.path.exists(path):
                MODELS[name] = load_model(path, compile=False)
                AVAILABLE_MODEL_NAMES.append(name)
                app.logger.info(f"Model '{name}' załadowany pomyślnie z '{path}'.")
            else:
                app.logger.warning(f"OSTRZEŻENIE: Plik modelu '{path}' dla '{name}' nie został znaleziony. Model nie będzie dostępny.")
        except Exception as e:
            app.logger.error(f"BŁĄD podczas ładowania modelu '{name}' z '{path}': {e}", exc_info=True)
    
    if not AVAILABLE_MODEL_NAMES:
        app.logger.warning("OSTRZEŻENIE: Żaden model nie został poprawnie załadowany. Aplikacja może nie działać zgodnie z oczekiwaniami.")
    else:
        app.logger.info(f"Załadowano następujące modele: {', '.join(AVAILABLE_MODEL_NAMES)}")

# Ładowanie modeli podczas uruchamiania aplikacji
load_application_models()

#  Funkcja do przetwarzania obrazu za pomocą modelu AI
def sharpen_image_ai(image_bytes, model_id):
    selected_model = MODELS.get(model_id)

    if selected_model is None:
        app.logger.error(f"Model '{model_id}' nie jest dostępny lub nie został załadowany.")
        raise ValueError(f"Model '{model_id}' nie jest załadowany lub nie istnieje.")

    # Dekodowanie bajtów obrazu do formatu OpenCV
    file_bytes_np = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Nie udało się zdekodować obrazu. Upewnij się, że to poprawny format obrazu (np. JPG, PNG).")

    img_float = img.astype('float32')

    # Dodanie wymiaru do tensoru dla modelu (batch dimension)
    input_tensor = np.expand_dims(img_float, axis=0)
    
    # Przetwarzanie obrazu modelem AI
    app.logger.info(f"Przetwarzanie obrazu (shape: {img.shape}) za pomocą modelu: {model_id}")
    output = selected_model.predict(input_tensor)[0]
    
    # Konwersja wyniku do formatu obrazu
    output = np.clip(output, 0, 255).astype('uint8')

    # Kodowanie przetworzonego obrazu do formatu JPG
    is_success, buffer = cv2.imencode(".jpg", output)
    if not is_success:
        raise ValueError("Nie udało się zakodować przetworzonego obrazu do formatu JPG.")
    return buffer.tobytes()

#  Główna strona aplikacji – ładowanie formularza z listą modeli
@app.route('/')
def index():
    default_model_id = None
    if AVAILABLE_MODEL_NAMES:
        if "Model 1" in AVAILABLE_MODEL_NAMES:
            default_model_id = "Model 1"
        else:
            default_model_id = AVAILABLE_MODEL_NAMES[0]
            
    return render_template('index.html', model_names=AVAILABLE_MODEL_NAMES, default_model_id=default_model_id)

#  Endpoint do przetwarzania obrazu wszystkimi modelami jednocześnie
@app.route('/process_all_models', methods=['POST'])
def process_all_models():
    if 'image' not in request.files:
        return jsonify({"error": "Brak pliku obrazu"}), 400
    
    image_file = request.files['image']
    original_image_bytes = image_file.read()

    if not original_image_bytes:
        return jsonify({"error": "Plik obrazu jest pusty."}), 400

    results = {}

    # Tworzenie unikalnej nazwy pliku i sesji
    original_image_filename = image_file.filename if image_file.filename else str(uuid.uuid4())
    safe_filename_part = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in original_image_filename)
    session_id = str(uuid.uuid4())
    PROCESSED_IMAGES_CACHE[session_id] = {}

    # Przetwarzanie obrazu dla każdego modelu
    for model_name in AVAILABLE_MODEL_NAMES:
        try:
            app.logger.info(f"Przetwarzanie obrazu dla sesji {session_id} za pomocą modelu: {model_name}")
            processed_bytes = sharpen_image_ai(original_image_bytes, model_name)
            
            # Tworzenie nazwy pliku wynikowego w cache
            cache_key = f"{safe_filename_part}_{model_name.replace(' ', '_')}.jpg"
            PROCESSED_IMAGES_CACHE[session_id][cache_key] = processed_bytes
            
            results[model_name] = {
                "url": f"/get_processed_image/{session_id}/{cache_key}",
                "status": "success"
            }
        except ValueError as e:
            app.logger.error(f"Błąd wartości (ValueError) podczas przetwarzania modelem {model_name} dla sesji {session_id}: {e}")
            results[model_name] = {"status": "error", "message": str(e)}
        except Exception as e:
            app.logger.error(f"Nieoczekiwany błąd podczas przetwarzania modelem {model_name} dla sesji {session_id}: {e}", exc_info=True)
            results[model_name] = {"status": "error", "message": "Wewnętrzny błąd serwera podczas przetwarzania."}
    
    return jsonify({
        "sessionId": session_id,
        "results": results
    })

#  Endpoint do pobrania przetworzonego obrazu z pamięci podręcznej
@app.route('/get_processed_image/<session_id>/<path:cache_key>')
def get_processed_image(session_id, cache_key):
    # Czyszczenie starego cache, jeśli przekroczono 50 sesji
    if len(PROCESSED_IMAGES_CACHE) > 50:
        keys_to_delete = list(PROCESSED_IMAGES_CACHE.keys())[:len(PROCESSED_IMAGES_CACHE)-50]
        for k_del in keys_to_delete:
            del PROCESSED_IMAGES_CACHE[k_del]
        app.logger.info(f"Cache cleanup: Usunięto {len(keys_to_delete)} starych sesji.")

    # Pobieranie przetworzonego obrazu z cache
    if session_id in PROCESSED_IMAGES_CACHE and cache_key in PROCESSED_IMAGES_CACHE[session_id]:
        image_bytes = PROCESSED_IMAGES_CACHE[session_id][cache_key]
        return send_file(
            io.BytesIO(image_bytes),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=cache_key # Sugerowana nazwa pliku przy zapisie
        )
    else:
        app.logger.warning(f"Nie znaleziono obrazu w cache dla session_id: {session_id}, cache_key: {cache_key}")
        return "Obraz nie znaleziony w cache", 404

#  Uruchamianie aplikacji na porcie 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
