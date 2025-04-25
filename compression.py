import matplotlib.pyplot as plt
import soundfile as sf
from scipy import fft
import numpy as np
import librosa
import argparse

# frequences des notes de musique
notes=[
    16.35,32.70,65.41,130.81,261.63,523.25,1046.50,2093.00,4186.01,8372.02,16744.04,
    17.33,34.65,69.30,138.59,277.18,554.37,1108.73,2217.46,4434.92,8869.84,17739.68,
    18.36,36.71,73.42,146.83,293.66,587.33,1174.66,2349.32,4698.64,9397.28,18794.56,
    19.45,38.89,77.78,155.56,311.13,622.25,1244.51,2489.02,4978.03,9956.06,19912.12,
    20.60,41.20,82.41,164.81,329.63,659.26,1318.51,2637.02,5274.04,10548.08,21096.16,
    21.83,43.65,87.31,174.61,349.23,698.46,1396.91,2793.83,5587.65,11175.30,22350.60,
    23.13,46.25,92.50,185.00,369.99,739.99,1479.98,2959.96,5919.91,11839.82,23679.64,
    24.50,49.00,98.00,196.00,392.00,783.99,1567.98,3135.96,6271.93,12543.86,25087.72,
    25.96,51.91,103.83,207.65,415.30,830.61,1661.22,3322.44,6644.88,13289.76,26579.52,
    27.50,55.00,110.00,220.00,440.00,880.00,1760.00,3520.00,7040.00,14080.00,28160.00,
    29.14,58.27,116.54,233.08,466.16,932.33,1864.66,3729.31,7458.62,14917.24,29834.48,
    30.87,61.74,123.47,246.94,493.88,987.77,1975.53,3951.07,7902.13,15804.26,31608.52
]
notes.sort()

def nom_de_la_note(frequence:float):
    """
    Renvoie le nom de la note correspondant à la fréquence donnée.
    """
    arrondie = frequence_note(frequence)
    if arrondie in notes:
        noms = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
        # On arrondit la frequence a la note la plus proche
        # On cherche l'index de la frequence dans le tableau notes
        nom = noms[notes.index(arrondie)%12]
        # On cherche l'octave de la note
        nom += str(int(notes.index(arrondie)/12)-1)
        return nom
    else:
        return ""


def frequence_note(vtest:float):
    if vtest in notes:
        return float(vtest)
    if vtest<min(notes):
        return float(vtest)
    if vtest>max(notes):
        return float(vtest)
    for f in range(len(notes)-1):
        if notes[f]<vtest<notes[f+1]:
            if abs(abs(notes[f]-vtest))==min(abs(notes[f]-vtest),abs(notes[f+1]-vtest)):
                return notes[f]
            else:
                return notes[f+1]

def find_nearest(array, value):
    """
    Renvoie l'index de la valeur la plus proche dans un tableau numpy.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def afficher_signaux(t:np.ndarray,signal1:np.ndarray,signal2:np.ndarray):
    """
    Affiche deux signaux sur le même graphique.
    """
    plt.figure("Signal")
    plt.subplot(211)
    plt.title("Signal d'origine")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.plot(t,signal1)
    plt.subplot(212)
    plt.title("Signal modifie")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.plot(t,signal2)

def afficher_frequences(X:np.ndarray,Y:np.ndarray,Y2:np.ndarray):
    """
    Affiche les frequences d'un signal
    """
    plt.figure("Fréquences")
    plt.subplot(211)
    plt.title("Frequences d'origines")
    plt.xlabel("frequence (Hz)")
    plt.ylabel("Amplitude")
#    plt.plot(X,librosa.amplitude_to_db(np.abs(Y)))
    plt.plot(X,np.abs(Y))
    plt.subplot(212)
    plt.title("Frequences modifiees")
    plt.xlabel("frequence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(X,np.abs(Y2))
#    plt.plot(X,librosa.amplitude_to_db(np.abs(Y2)))

def afficher_spectrogramme(data1:np.ndarray,data2:np.ndarray,samp_fichier:int):
    """
    Affiche le spectrogramme d'un signal
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(data1,sr=samp_fichier,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
    f1, voiced_flag2, voiced_probs2 = librosa.pyin(data2,sr=samp_fichier,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
    # On recupere les temps de chaque echantillon
    times0 = librosa.times_like(f0, sr=samp_fichier)
    times1 = librosa.times_like(f1, sr=samp_fichier)

    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(data1)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(data2)), ref=np.max)

    # On affiche le spectrogramme
    plt.figure("Spectrogramme")

    plt.subplot(211)
    plt.title("Spectrogramme d'origine")
    librosa.display.specshow(D1, x_axis='time', y_axis='log', sr=samp_fichier)
    plt.colorbar(format='%+2.0f dB')
    plt.plot(times0, f0, label='frequence fondamentale', color='cyan')
    plt.legend(loc='upper right')


    plt.subplot(212)
    plt.title("Spectrogramme modifie")
    librosa.display.specshow(D2, x_axis='time', y_axis='log', sr=samp_fichier)
    plt.colorbar(format='%+2.0f dB')
    plt.plot(times1, f1, label='frequence fondamentale', color='cyan')
    plt.legend(loc='upper right')

def psnr(audio_original,audio_compressed):
    mse = np.mean((audio_original - audio_compressed) ** 2)  # Calculer l'erreur quadratique moyenne (MSE)
    # Calculer le PSNR
    # La valeur maximale de l'audio est 1 car le type de l'audio est float64
    R = 1.0
    if mse == 0:
        psnrcalc = float('inf')  # Si MSE est 0, PSNR est infini (pas de perte)
    else:
        psnrcalc = 10 * np.log10((R ** 2) / mse)

    return psnrcalc

def futures_frequences(data:np.ndarray,samp_fichier:int,nombre_de_freq,graphique=False):
    "Simplifie le signal en arrondissant les frequences a la note la plus proche"

    N = len(data)
    X = fft.rfftfreq(N, 1 / (samp_fichier))
    Y = fft.rfft(data)
    #abs prend le module des valeurs de la fft

    # amplitudes_retenues correspond aux index des frequences les plus importantes
    amplitudes_retenues=np.array([],dtype=np.int64)
    # On cherche les index des frequences les plus importantes
    for i in range(1,1+nombre_de_freq):
        # np.sort(np.abs(Y))[-i] prend la ième valeur la plus grande de Y
        nouvelle_amplitude = np.where(np.isclose(np.abs(Y),np.sort(np.abs(Y))[-i]))[0]
        amplitudes_retenues = np.concatenate((amplitudes_retenues,nouvelle_amplitude))
    # freq_importantes correspond aux frequences retenues (par exemple 440Hz)
    freq_importantes=X[amplitudes_retenues]
    # volume_des_freq_importantes correspond aux amplitudes des frequences retenues
    volume_des_freq_importantes=Y[amplitudes_retenues]


    notes=[]

    Y2=np.zeros(len(Y), dtype=complex)
    for gd in freq_importantes:
        # On arrondit la frequence a la note la plus proche

        # L'oreille humaine ne percoit pas les frequences en dessous de 20Hz et au dessus de 20kHz
        # On ne garde que les frequences entre 20Hz et 20kHz
        if gd < 20 or gd > 20000:
            continue
        # On arrondit la frequence a la note la plus proche
        f=frequence_note(gd)
        # On recupere le nom de la note
        note=nom_de_la_note(gd)
        # On ajoute la note a la liste des notes
        try:
            if note not in notes:
                notes.append(note)
        except:
            notes.append(note)
        # On recupere l'amplitude de la frequence
        y=volume_des_freq_importantes[np.where(np.isclose(freq_importantes,gd))]
        # On cherche la position de la frequence dans le signal modifie
        position = find_nearest(X, f)
        # On place la frequence dans le signal modifie
        Y2[position]=y[0]



    # On recrée un signal a partir de la fft modifiee
    # On utilise la transformée de Fourier inverse pour recréer le signal
    signal_modifie = fft.irfft(Y2)

    if graphique == True:
        afficher_frequences(X/samp_fichier,Y,Y2)
    # On affiche les notes retenues
    print(notes)
    return signal_modifie.astype(np.float32)


def main(file:str,ms=50,nombre_de_freq=3,graphique=False):
    """
    Fonction principale qui lit un fichier audio, le traite et affiche les graphiques.
    """
    # On lit le fichier audio
    data,samplerate = sf.read(file)
    # On crée un tableau de temps pour l'affichage
    t = np.arange(len(data)) / samplerate
    # On crée un tableau vide pour stocker le signal modifie
    s=np.array([],dtype=np.float32)

    # On calcule le nombre de points a traiter par sous-signal
    points=int((ms*np.where(np.isclose(t,1.0))[0][0]/1000)/2)*2

    for i in range(int(len(data)/points)):
        # On traite le signal en morceaux de points
        if i == int(int(len(data)/points)/2) and graphique == True:
            sous_graphique=True
        else:
            sous_graphique=False
        # On extrait le sous-signal
        # range_start et range_end sont les index de debut et de fin du sous-signal
        range_start = int(points*i)
        range_end = int(points*i+points)
        sous_data=data[range_start:range_end]
        # On traite le sous-signal
        s2=futures_frequences(sous_data,samplerate,nombre_de_freq=nombre_de_freq,graphique=sous_graphique)

        s=np.concatenate((s,s2))
        if graphique == True:
            afficher_signaux(t[range_start:range_end],sous_data,s2)

    # On traite le dernier sous-signal (le fichier audio n'est pas toujours divisible par ms millisecondes)
    points_fin=int(len(data)%points)
    sous_data=data[len(data)-points_fin:]
    try:
        s2=futures_frequences(sous_data,samplerate,nombre_de_freq=nombre_de_freq,graphique=graphique)
    except:
        s2=np.array([],dtype=np.float32)
    s=np.concatenate((s,s2))



    # On recrée le signal audio a partir de la fft modifiee

    print("\nLa PSNR entre les fichiers audio est :",round(psnr(data,s),2),"dB\nPSNR élevée = fichiers qui se ressemblent :")
    print("PNSR inférieur à 20 dB = perte de qualité importante")
    print("PNSR compris entre 20 et 40 dB = perte de qualité acceptable")
    print("PNSR compris entre 40 et 50 dB = perte de qualité imperceptible")
    print("PNSR supérieur à 50 dB = fichiers identiques\n")
    print("Le fichier audio modifié a été enregistré sous le nom 'reconstructed.wav'")

    sf.write('reconstructed.wav',np.real(s).astype(np.float32),samplerate)
    if graphique:
        afficher_spectrogramme(data,s,samplerate)
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='Le fichier a traiter, doit etre en .wav, en mono et dans le dossier du programme')
parser.add_argument('--graphique', action='store_false', help='Desactive les graphiques')
parser.add_argument('--ms', type=int, default=50, help='Le temps en ms entre chaque echantillon dont on calcule la fft.Par default egal a 50ms')
parser.add_argument('--freq', type=int, default=3, help='Le nombre de frequences retenues. Par default egal a 3')
args = parser.parse_args()

main(args.file, ms=args.ms, nombre_de_freq=args.freq, graphique=args.graphique)



# Exemple de test:
# python compression.py la-440.wav --graphique --ms 50 --freq 3
