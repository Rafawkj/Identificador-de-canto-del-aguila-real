import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Función para aplicar un filtro pasa banda
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs             #mitad de la frecuencia de muestreo. 11025 
    low = lowcut / nyquist         #La frecuencia de muestreo debe ser mayor o 
    high = highcut / nyquist       #igual a 2 veces la frecuencia máxima de la señal
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Cargar los audios
def cargar_audio(ruta, sr=22050): # sr: Frecuencia de muestreo deseada (Hz). Por defecto, es 22050 Hz.
    audio, sr = librosa.load(ruta, sr=sr)
    return audio, sr              # Devuelve el audio cargado (como un arreglo de datos) y su frecuencia de muestreo.

# Calcular la Transformada de Fourier
def calcular_fft(audio, sr):
    n = len(audio)
    fft = np.fft.fft(audio)                    # Esto transforma la señal del dominio del tiempo al dominio de la frecuencia.
    fft_magnitude = np.abs(fft[:n // 2])
    freqs = np.fft.fftfreq(n, 1 / sr)[:n // 2] # Calcular las frecuencias correspondientes al espectro.
    return freqs, fft_magnitude

# Comparar espectros
def comparar_espectros(fft_modelo, fft_ambiental, umbral=0.7):
    n = min(len(fft_modelo), len(fft_ambiental))                 # Tamaño mínimo
    fft_modelo = fft_modelo[:n]
    fft_ambiental = fft_ambiental[:n]
    correlacion = np.corrcoef(fft_modelo, fft_ambiental)[0, 1]   # Calcular el coeficiente de correlación entre los dos espectros
    return correlacion > umbral, correlacion 
#La correlación de Pearson es una medida estadística que 
#indica el grado de relación lineal entre dos conjuntos de datos.

# Procesamiento principal
def detectar_aguila(ruta_modelo, ruta_ambiental, lowcut, highcut):
    # Cargar los audios
    modelo, sr_modelo = cargar_audio(ruta_modelo)
    ambiental, sr_ambiental = cargar_audio(ruta_ambiental)
    
    # Filtrar el audio ambiental
    ambiental_filtrado = butter_bandpass_filter(ambiental, lowcut, highcut, sr_ambiental)
    
    # Calcular las FFT
    freqs_modelo, fft_modelo = calcular_fft(modelo, sr_modelo)
    freqs_ambiental, fft_ambiental = calcular_fft(ambiental_filtrado, sr_ambiental)
    
    # Comparar espectros
    detectado, correlacion = comparar_espectros(fft_modelo, fft_ambiental)
    
    # Resultados
    if detectado:
        print(f"Canto del águila detectado. Correlación: {correlacion:.2f}")
    else:
        print(f"Canto del águila NO detectado. Correlación: {correlacion:.2f}")
    
    # Graficar espectros (Transformada de Fourier)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs_modelo, fft_modelo, label='Modelo (Canto del Águila)')
    plt.title('Espectro del Modelo')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs_ambiental, fft_ambiental, label='Audio Ambiental Filtrado')
    plt.title('Espectro del Audio Ambiental Filtrado')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Graficar espectrogramas
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    modelo_spec = librosa.amplitude_to_db(np.abs(librosa.stft(modelo)), ref=np.max)
    librosa.display.specshow(modelo_spec, sr=sr_modelo, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma del Modelo (Canto del Águila)')
    
    plt.subplot(2, 1, 2)
    ambiental_spec = librosa.amplitude_to_db(np.abs(librosa.stft(ambiental_filtrado)), ref=np.max)
    librosa.display.specshow(ambiental_spec, sr=sr_ambiental, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma del Audio Ambiental Filtrado')
    
    plt.tight_layout()
    plt.show()

# Parámetros del filtro (ajusta según el canto del águila)
lowcut = 500  # Frecuencia mínima en Hz
highcut = 4000  # Frecuencia máxima en Hz

# Rutas de los archivos de audio
ruta_modelo = r'C:\Users\Rafael\Downloads\Noveno Semestre\Analisis de señales\Proyecto_final\audio_aguila.wav' #audio_aguila_selva 
ruta_ambiental = r'C:\Users\Rafael\Downloads\Noveno Semestre\Analisis de señales\Proyecto_final\audio_aguila_selva.wav'# ó aguila_pescadora

# Llamada a la función principal
detectar_aguila(ruta_modelo, ruta_ambiental, lowcut, highcut)