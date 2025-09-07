from matplotlib import pyplot as plt
from scipy import io
import numpy as np
import argparse

def hann_window(N):
    """Génère une fenêtre de Hann."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N))
def hamming_window(N):
    """Fenêtre de Hamming : similaire à Hann mais avec une forme légèrement différente"""
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
def rect_window(N):
    """Fenêtre rectangulaire : pas d'atténuation, brut (moins recommandé en général)"""
    return np.ones(N)


class Note:
    """
    Gère les notes de musique et leurs fréquences.
    """
    def __init__(self):
        """Fréquences des notes de musique (en Hz)"""
        self.notes=[
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
        self.notes.sort()
    def nom_de_la_note(self,frequence:float):
        """
        Renvoie le nom de la note correspondant à la fréquence donnée.
        """
        arrondie = self.frequence_note(frequence)
        if arrondie in self.notes:
            noms = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
            # On arrondit la frequence a la note la plus proche
            # On cherche l'index de la frequence dans le tableau notes
            nom = noms[self.notes.index(arrondie)%12]
            # On cherche l'octave de la note
            nom += str(int(self.notes.index(arrondie)/12)-1)
            return nom
        else:
            return ""
    def frequence_note(self,vtest:float):
        """
        Renvoie la fréquence de la note la plus proche de la fréquence donnée.
        Si la fréquence est en dehors de la plage des notes, renvoie la fréquence elle-même.
        """
        if vtest in self.notes:
            return float(vtest)
        if vtest<min(self.notes):
            return float(vtest)
        if vtest>max(self.notes):
            return float(vtest)
        for f in range(len(self.notes)-1):
            if self.notes[f]<vtest<self.notes[f+1]:
                if abs(abs(self.notes[f]-vtest))==min(abs(self.notes[f]-vtest),abs(self.notes[f+1]-vtest)):
                    return self.notes[f]
                else:
                    return self.notes[f+1]
    def find_nearest(self,array, value):
        """
        Renvoie l'index de la valeur la plus proche dans un tableau numpy.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    def justifieur(self,stft_matrix,samplerate,frame_size,nombre_de_freq):
        """Ajuste les fréquences dans la matrice STFT aux notes de musique les plus proches.
         Conserve uniquement un nombre spécifié de fréquences par trame.
         C'est cette fonction qui traite le signal."""
        #On retourne la matrice STFT modifiée pour avoir des lignes correspondant à une fft
        data=np.transpose(stft_matrix)
        data2=np.zeros(data.shape,dtype=data.dtype)

        for ind in range(len(data)):
            fft=data[ind]
            if min(fft)==max(fft): #S'il n'y a que des 0
                continue

            # X correspond aux frequences en Hz, c'est l'axe des absicces du spectrogramme
            X = np.fft.rfftfreq(frame_size, 1 / (samplerate))
            

            # amplitudes_retenues correspond aux index des frequences les plus importantes
            amplitudes_retenues=np.array([],dtype=np.int64)
            # On cherche les index des frequences les plus importantes
            for i in range(1,1+nombre_de_freq):
                # nouvelle_amplitude est l'index de la i-eme frequence la plus importante
                nouvelle_amplitude = np.where(np.isclose(np.abs(fft),np.sort(np.abs(fft))[-i]))[0]
                amplitudes_retenues = np.concatenate((amplitudes_retenues,nouvelle_amplitude))
            # freq_importantes correspond aux frequences retenues (par exemple 440Hz)
            freq_importantes=X[amplitudes_retenues]
            # volume_des_freq_importantes correspond aux amplitudes des frequences retenues
            volume_des_freq_importantes=fft[amplitudes_retenues]
            # On cree un tableau de la meme taille que fft mais avec uniquement les frequences arrondies aux notes
            notes=[]

            fft_arrondie =np.zeros(
                len(fft), 
                dtype=np.complex64)

            for gd in freq_importantes:
                if gd < 20 or gd > 20000:
                    continue
                # On arrondit la frequence a la note la plus proche
                f=self.frequence_note(gd)
                # On recupere le nom de la note
                note=self.nom_de_la_note(gd)
                # On ajoute la note a la liste des notes
                try:
                    if note not in notes:
                        notes.append(note)
                except:
                    notes.append(note)
                # On recupere l'amplitude de la frequence
                y=volume_des_freq_importantes[np.where(np.isclose(freq_importantes,gd))]
                # On cherche la position de la frequence dans le signal modifie
                position = self.find_nearest(X, f)
                # On place la frequence dans le signal modifie

                fft_arrondie[position]=y[0]
            print(notes)
            data2[ind]=fft_arrondie
        return np.transpose(data2)



class FFT:
    """Gère la STFT et l'ISTFT 
    La transformée de Fourier permet de passer du domaine temporel au domaine fréquentiel.
    La STFT (Short-Time Fourier Transform) permet d'analyser les variations fréquentielles
    d'un signal au cours du temps en découpant le signal en trames temporelles.
    L'ISTFT (Inverse Short-Time Fourier Transform) permet de reconstruire le signal temporel
    à partir de sa représentation fréquentielle."""
    def __init__(self,frame_size,hop_size,window_fn=hann_window):
        """Initialise les paramètres de la STFT/ISTFT : 
        - frame_size : taille de la fenêtre
        - hop_size : taille du saut entre les fenêtres
        - window_fn : fonction de fenêtre (par défaut Hann)"""
        self.frame_size=frame_size
        self.hop_size=hop_size
        self.window_fn=window_fn
    def fft(self,x):
        """
        FFT récursive de Cooley-Tukey (pour longueur N=2^m)
        La FFT (Fast Fourier Transform) est une version optimisée de la transformée de Fourier.
        """
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        if N <= 1:
            return x
        if N % 2 != 0:
            raise ValueError("La longueur doit être une puissance de 2")
        
        X_even = self.fft(x[::2])
        X_odd  = self.fft(x[1::2])
        
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        X = np.zeros(N, dtype=complex)
        half_N = N//2
        X[:half_N] = X_even + factor[:half_N] * X_odd
        X[half_N:] = X_even - factor[:half_N] * X_odd
        return X
    def rfft(self,x):
        """
        RFFT manuelle pour signal réel
        Renvoie N/2+1 coefficients
        La RFFT représente la moitié des coefficients de la FFT pour les signaux réels,
        car la seconde moitié est redondante (conjuguée de la première moitié).
        """
        X_full = self.fft(x)
        N = len(x)
        return X_full[:N//2 + 1]

    def irfft(self,X_rfft):
        """
        iRFFT manuelle à partir des coefficients RFFT.
        X_rfft : N/2+1 coefficients complexes
        Renvoie un signal réel.
        """
        N = (len(X_rfft)-1)*2  # longueur du signal original
        X_full = np.zeros(N, dtype=complex)
        
        # Parties positive et Nyquist
        X_full[:len(X_rfft)] = X_rfft
        
        # Parties négative (symétrie conjuguée)
        X_full[len(X_rfft):] = np.conj(X_rfft[1:-1][::-1])
        
        # iFFT via Cooley-Tukey inverse
        x = self.ifft_ct(X_full)
        
        return np.real(x)

    def ifft_ct(self,X):
        """
        iFFT via Cooley-Tukey
        """
        N = len(X)
        X_conj = np.conj(X)
        x = self.fft(X_conj)  # FFT sur le conjugué
        return np.conj(x)/N

    def stft(self,signal):
        """Applique la STFT au signal donné.
        Retourne une matrice STFT (fréquences x trames)"""
        window = self.window_fn(self.frame_size)
        num_frames = 1 + (len(signal) - self.frame_size) // self.hop_size
        stft_result = []

        for i in range(num_frames):
            start = i * self.hop_size
            frame = signal[start:start + self.frame_size]
            windowed = frame * window

            #Si vous voulez utiliser mon implémentation de la FFT, décommentez la ligne suivante et commentez la ligne après
            #Cela vous prouvera la supériorité du C sur Python en termes de performance... (np.fft.rfft est en C)
            #spectrum = self.rfft(windowed) 
            spectrum = np.fft.rfft(windowed)

            stft_result.append(spectrum)

        return np.transpose(np.array(stft_result))
    def istft(self,stft_matrix):
        """Reconstruit le signal à partir de la matrice STFT donnée."""
        window = self.window_fn(self.frame_size)
        num_frames = stft_matrix.shape[1]
        output_length = self.frame_size + (num_frames - 1) * self.hop_size
        output_signal = np.zeros(output_length)
        window_sums = np.zeros(output_length)

        for i in range(num_frames):
            start = i * self.hop_size
            spectrum = np.transpose(stft_matrix)[i]
            #Si vous voulez utiliser mon implémentation de la iFFT, décommentez la ligne suivante et commentez la ligne après
            #frame = self.irfft(spectrum)
            frame = np.fft.irfft(spectrum)
            output_signal[start:start + self.frame_size] += frame * window
            window_sums[start:start + self.frame_size] += window ** 2

        # Normalisation
        nonzero = window_sums > 1e-10
        output_signal[nonzero] /= window_sums[nonzero]
        return output_signal


def psnr(audio_original,audio_compressed):
    """Calcule le PSNR entre deux signaux audio.
    Le PSNR (Peak Signal-to-Noise Ratio) est une mesure de la qualité de
    reconstruction d'un signal compressé par rapport à l'original.
    Plus le PSNR est élevé, meilleure est la qualité de reconstruction."""
    # On s'assure que les deux signaux ont la même longueur
    audio_original = audio_original[:min(len(audio_original),len(audio_compressed))]
    audio_compressed = audio_compressed[:min(len(audio_original),len(audio_compressed))]
    mse = np.mean((audio_original - audio_compressed) ** 2)  # Calculer l'erreur quadratique moyenne (MSE)
    # La valeur maximale de l'audio est 1 car le type de l'audio est float64
    R = 1.0
    if mse == 0:
        psnrcalc = float('inf')  # Si MSE est 0, PSNR est infini (pas de perte)
    else:
        psnrcalc = 10 * np.log10((R ** 2) / mse)

    return psnrcalc

def load_audio_file(file:str):
    """Charge un fichier audio et le convertit en mono float32.
    Retourne le taux d'échantillonnage et les données audio."""
    samplerate, data = io.wavfile.read(file)

    # Convertir dans [-1, 1] selon le format d'origine
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    elif data.dtype == np.float32:
        pass
    else:
        raise ValueError("Le fichier audio doit être au format .wav")
    # Si le fichier est stéréo, on convertit en mono en moyennant les canaux
    if data.ndim == 2 and data.shape[1] == 2:
        data = np.mean(data, axis=1)  # Stéréo → Mono
    data=np.clip(data,-1,1)
    return samplerate,data
def save_audio_file(data:np.ndarray,samplerate,file:str):
    """Sauvegarde un signal audio float32 dans un fichier .wav"""
    io.wavfile.write(file,samplerate,data)

class Graphiques:
    def __init__(self):
        """Initialisation de la classe Graphiques"""
        pass
    def spectrogrammes(self,stft1,stft2,t1,t2):
        """Affiche les spectrogrammes de deux matrices STFT."""
        plt.figure("Spectrogramme original")
        plt.imshow(10 * np.log10(np.abs(stft1+1e-7)),origin='lower', aspect='auto',extent=[0, t1[-1], 20, 20000])
        plt.ylabel("Fréquence (Hz)")
        plt.xlabel("Temps (s)")
        plt.colorbar(label="Amplitude (dB)")

        plt.figure("Spectogramme modifié")
        plt.imshow(10 * np.log10(np.abs(stft2)+1e-7),origin='lower', aspect='auto',extent=[0, t2[-1], 20, 20000])
        plt.ylabel("Fréquence (Hz)")
        plt.colorbar(label="Amplitude(dB)")
    def signaux(self,s1,s2,t1,t2):
        """Affiche deux signaux temporels."""
        plt.figure("Signal d'origine")
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")
        #plt.axis((0,max(t1),-1,1))
        plt.plot(t1,s1)
        plt.figure("Signal modifie")
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")
        #plt.axis((0,max(t1),-1,1))

        plt.plot(t2,s2)


def main(file:str,frame_size, hop_size, nombre_de_freq):
    """
    Fonction principale qui lit un fichier audio, le traite et affiche les graphiques.
    """
    # On lit le fichier audio
    samplerate, data = load_audio_file(file)



    pad_width = frame_size  # 1 frame de chaque côté
    data_padded = np.pad(data, (pad_width, pad_width), mode="constant")

    fft=FFT(frame_size,hop_size)
    note=Note()
    graphique=Graphiques()
    stft_matrix=fft.stft(data_padded)
    stft_2=note.justifieur(stft_matrix,samplerate,frame_size,nombre_de_freq)
    data2_padded=fft.istft(stft_2).astype(np.float32)

    data2 = data2_padded[pad_width:-pad_width]

    coeff=max(abs(data))/max(abs(data2))
    data2*=coeff
    save_audio_file(data2,samplerate,"reconstructed.wav")

    t1 = np.linspace(0, len(data) / samplerate, len(data))
    t2 = np.linspace(0,len(data2)/samplerate,len(data2))
    verif=fft.stft(data2)
    #print(verif[0])
    graphique.spectrogrammes(verif,stft_2,t1,t2)
    graphique.signaux(data,data2,t1,t2)

    plt.show()
#main("sounds/SNCF.wav",2048,512,1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traitement audio avec STFT et compression (sous la forme sounds/fichier.wav).")
    parser.add_argument("file", type=str, help="Chemin vers le fichier audio .wav")
    parser.add_argument("--frame_size", type=int, default=2048, help="Taille de la fenêtre pour la STFT")
    parser.add_argument("--hop_size", type=int, default=512, help="Taille du saut entre les fenêtres")
    parser.add_argument("--nombre_de_freq", type=int, default=3, help="Nombre de fréquences à conserver par trame")

    args = parser.parse_args()

    main(args.file, args.frame_size, args.hop_size, args.nombre_de_freq)
    #
