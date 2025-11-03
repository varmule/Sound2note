"""
App which convert a sound to notes
"""

import toga
from toga import paths
from toga.colors import rgb
from toga.style.pack import COLUMN, ROW, CENTER, Pack
from pathlib import Path
import webbrowser
from sound2note import compression_stft

import numpy as np
import asyncio

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Évite Tkinter pour le conflits avec toga

try:
    import miniaudio
    miniaudio_available = True
except ImportError:
    import sounddevice as sd
    miniaudio_available = False


class FigureWindow(toga.Window):
    """Fenêtre pour afficher une figure matplotlib avec zoom et navigation."""

    def __init__(self, fig, title="Graphique"):
        super().__init__(title=title, size=(800, 600))
        self.fig = fig

        # Créé un buffer pour sauvegarder l'image matplotlib
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        self.original_image = Image.open(buf)

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Boite principale
        self.main_box = toga.Box(style=Pack(flex=1, direction=COLUMN))

        # Image View pour afficher l'image
        self.image_view = toga.ImageView(style=Pack(flex=1))
        self.main_box.add(self.image_view)

        # Première ligne de boutons : Zoom
        zoom_box = toga.Box(style=Pack(direction=ROW, justify_content=CENTER, margin=5, gap=5))
        self.btn_zoom_in = toga.Button("+", on_press=self.zoom_in, style=Pack(flex=0.5))
        self.btn_zoom_out = toga.Button("-", on_press=self.zoom_out, style=Pack(flex=0.5))
        zoom_box.add(self.btn_zoom_in)
        zoom_box.add(self.btn_zoom_out)

        # Seconde ligne : Les flèches pour se déplacer sur le graphique
        arrow_box = toga.Box(style=Pack(direction=ROW, justify_content=CENTER, margin=5, gap=5))
        self.btn_up = toga.Button("↑", on_press=lambda w: self.pan(0, -50), style=Pack(flex=0.25))
        self.btn_down = toga.Button("↓", on_press=lambda w: self.pan(0, 50), style=Pack(flex=0.25))
        self.btn_left = toga.Button("←", on_press=lambda w: self.pan(-50, 0), style=Pack(flex=0.25))
        self.btn_right = toga.Button("→", on_press=lambda w: self.pan(50, 0), style=Pack(flex=0.25))

        # Pour l’ordre naturel: ↑ ↓ ← →
        arrow_box.add(self.btn_up)
        arrow_box.add(self.btn_down)
        arrow_box.add(self.btn_left)
        arrow_box.add(self.btn_right)

        self.main_box.add(zoom_box)
        self.main_box.add(arrow_box)
        self.content = self.main_box

        self.update_image()
        self.show()

    def update_image(self):

        img=self.original_image
        # Taille de la fenêtre
        win_w, win_h = map(int, self.size)  # largeur et hauteur de la fenêtre

        # Calcul crop selon zoom et offset
        img_w, img_h = img.size
        crop_w = int(win_w / self.scale)
        crop_h = int(win_h / self.scale)

        # Limiter crop à l'image
        crop_w = min(crop_w, img_w)
        crop_h = min(crop_h, img_h)

        # Centre + offset
        center_x = img_w // 2 + int(self.offset_x)
        center_y = img_h // 2 + int(self.offset_y)

        left = max(0, center_x - crop_w // 2)
        upper = max(0, center_y - crop_h // 2)
        right = min(img_w, left + crop_w)
        lower = min(img_h, upper + crop_h)

        cropped = img.crop((left, upper, right, lower))

        # Redimensionner pour remplir la fenêtre
        resized = cropped.resize((win_w, win_h), Image.LANCZOS)

        # Convertir en bytes pour ImageView
        buf2 = BytesIO()
        resized.save(buf2, format='PNG')
        buf2.seek(0)
        self.image_view.image = toga.Image(src=buf2.read())


    # ---- Zoom ----
    def zoom_in(self, widget):
        """Permet de zoomer l'image"""
        self.scale *= 1.3
        self.update_image()

    def zoom_out(self, widget):
        "Permet de dézoomer"
        if self.scale > 1.0:
            self.scale /= 1.3
            self.update_image()

    # ---- Pan ----
    def pan(self, dx, dy):
        "Permet de déplacer l'image"
        self.offset_x += dx / self.scale
        self.offset_y += dy / self.scale
        self.update_image()


def show_figure(fig, title="Graphique"):
    "Affiche la figure matplotlib"
    FigureWindow(fig, title=title)

def plot_signals(signal, signal_reconstructed, sample_rate):
    "Affiche les signaux"

    # On créé les axes temporels
    t1 = np.arange(len(signal)) / sample_rate
    t2 = np.arange(len(signal_reconstructed)) / sample_rate

    # Signal original
    fig1, ax1 = plt.subplots()
    ax1.plot(t1, signal)
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Amplitude")
    show_figure(fig1, "Signal d'origine")

    # Signal modifié
    fig2, ax2 = plt.subplots()
    ax2.plot(t2, signal_reconstructed)
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Amplitude")
    show_figure(fig2, "Signal modifié")


def plot_spectrograms(stft_matrix, stft_matrix_reconstructed, sample_rate, frame_size, hop_size):
    "Permet d'afficher les spectrogrammes à partir des paramêtres d'une stft"
    # f correspond au domaine fréquentiel, il stocke les fréquences en Hertz.
    f = np.fft.rfftfreq(frame_size, 1 / sample_rate)

    # t correspondent au domain temporel. Ils sont souvent légèrement différents
    t1 = np.arange(stft_matrix.shape[1]) * hop_size / sample_rate
    t2 = np.arange(stft_matrix_reconstructed.shape[1]) * hop_size / sample_rate

    # Spectrogramme original
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(10 * np.log10(np.abs(stft_matrix) + 1e-7), origin='lower', aspect='auto',
                     extent=[t1[0], t1[-1], f[0], f[-1]/3])
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Fréquence (Hz)")
    fig1.colorbar(im1, ax=ax1, label="Amplitude (dB)")
    show_figure(fig1, "Spectrogramme original")

    # Spectrogramme modifié
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(10 * np.log10(np.abs(stft_matrix_reconstructed) + 1e-7), origin='lower', aspect='auto',
                     extent=[t2[0], t2[-1], f[0], f[-1]/3])
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Fréquence (Hz)")
    fig2.colorbar(im2, ax=ax2, label="Amplitude (dB)")
    show_figure(fig2, "Spectrogramme modifié")


def plot_notes(notes, hop_size, sample_rate, nombre_de_freq):
    "Montre les notes de musique sous forme de piano roll"

    # Si vous essayez de comprendre le code,
    # ne vous attardez pas sur cette fonction, elle ne sert quasiment pas
    # et est mal documentée
    fig, ax = plt.subplots()

    #On recréé le noms des notes qu'on veut
    noms_notes = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
    octaves = range(2, 6)
    labels = [f"{n}{o}" for o in octaves for n in noms_notes]
    n_notes = len(labels)
    n_trames = len(notes)
    matrice = np.zeros((n_notes, n_trames), dtype=np.float32)
    for t, trame in enumerate(notes):
        if not trame:
            continue
        for i, note_nom in enumerate(trame[:nombre_de_freq]):
            if note_nom not in labels:
                continue
            intensite = 1.0 - i / max(1, nombre_de_freq - 1) if nombre_de_freq > 1 else 1.0
            matrice[labels.index(note_nom), t] = intensite
    # Lissage des notes courtes
    min_frames = int(0.08 * sample_rate / hop_size)
    for i in range(n_notes):
        active = matrice[i, :] > 0
        start = None
        for t in range(n_trames):
            if active[t] and start is None:
                start = t
            elif not active[t] and start is not None:
                if t - start < min_frames:
                    matrice[i, start:t] = 0
                start = None
    # Lissage temporel à l'aide de la fenêtre de hanning
    window_size = max(1, int(0.02 * sample_rate / hop_size))
    if window_size > 1:
        window = np.hanning(window_size)
        window /= np.sum(window)
        for i in range(n_notes):
            matrice[i, :] = np.convolve(matrice[i, :], window, mode="same")
    if matrice.max() > 0:
        matrice /= matrice.max()
    t = np.arange(n_trames) * hop_size / sample_rate

    image=ax.imshow(matrice,origin="lower",aspect="auto",extent=[t[0], t[-1], 0, n_notes],cmap="YlGnBu",interpolation="nearest")

    for i, n in enumerate(labels):
        if "#" in n:  # dièse → ligne fine et pâle
            ax.axhline(i, color="black", lw=0.3, ls="--", alpha=0.15)
        else:  # note naturelle → ligne plus visible
            ax.axhline(i, color="black", lw=0.5, ls="-", alpha=0.4)

    # On n'affiche que certaines notes sur l'axe Y pour que ce soit plus lisible
    positions_naturelles = [i for i, n in enumerate(labels) if (n[:-1]in["Sol","Do","Mi"])]
    etiquettes_naturelles = [labels[i] for i in positions_naturelles]
    ax.set_yticks(positions_naturelles, etiquettes_naturelles)

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Notes")
    ax.set_title("Piano Roll")
    fig.colorbar(image, ax=ax, label="Amplitude")
    show_figure(fig, "Piano Roll")


# Fonction audio mémoire pour miniaudio.
# C'est un générateur, permettant de lire des sons
if miniaudio_available:
    def memory_stream(npa: np.ndarray):
        # Initialisation
        required_frames = yield b""
        frames = 0
        while frames < len(npa):
            frames_end = frames + required_frames
            # On retourne des petits bouts à chaque fois
            required_frames = yield npa[frames:frames_end]
            frames = frames_end

# ------------------- Application principale -------------------
class Sound2note(toga.App):
    """Application Toga pour convertir un son en notes musicales."""

    def startup(self):
        self.main_box = toga.Box()

        # Variables audio/STFT
        self.frame_size = 1024 # Taille de la fenêtre. Plus elle est grande, meilleure est la qualité
        self.hop_size = 256 # Taille de saut entre chaque fenêtre. Plus elle est petite, meilleure est la qualité
        self.nombre_de_freq = 3 # Nombre de fréquences conservées par fenêtre. Logiquement, plus ce nombre est grand, meilleure est la qualité
        self.file_path = None # Chemin du fichier audio chargé
        self.recording = False # Indique si l'enregistrement est en cours
        self.signal = None # Signal audio original
        self.sample_rate = None # Fréquence d'échantillonnage du signal
        self.signal_reconstructed = None # Signal audio reconstruit à partir de notes de musique
        self.stft_matrix = None # Matrice STFT (Short Time Fourier Transform) du signal original
        self.stft_matrix_reconstructed = None # Matrice STFT du signal reconstruit, modifiée pour ne garder que certaines fréquences
        self.notes = None # Liste des notes de musique extraites du signal

        # Buffer pour l'enregistrement audio
        self.buffer_chunks = []

        # Initialisation de la fenêtre principale
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = self.main_box
        self.setup_ui()
        self.main_window.show()

    def setup_ui(self):
        """Configure l'interface utilisateur."""
        # Cette méthode crée les différents éléments de l'interface utilisateur
        # Elle est longue et inutile à commenter en détail
        # En gros, on met des boutons et textes dans des boites qu'on imbrique entre elles
        self.traitement_audio = toga.Box(style=Pack(direction=COLUMN, align_items=CENTER, margin=10, flex=1))

        self.audio = toga.Box(style=Pack(direction=ROW, gap=60,margin=5), children=[
            toga.Button("Charger un fichier audio", on_press=self.load_audio, style=Pack(margin=5, flex=0.5, margin_left=30)),
            toga.Button("Enregistrer un audio", on_press=self.record_audio, style=Pack(margin=5, flex=0.5, margin_right=30))
        ])

        self.label_audio = toga.Label("Aucun fichier sélectionné", style=Pack(margin=5, text_align=CENTER))
 
        self.parametres = toga.Box(style=Pack(direction=COLUMN, align_items=CENTER,margin=5), children=[
            toga.Box(style=Pack(direction=ROW, align_items=CENTER), children=[
                toga.Label("Taille de la fenêtre (frame size): ", style=Pack(text_align=CENTER, flex=0.5, margin=10)),
                toga.NumberInput(value=1024, style=Pack(flex=0.5, margin=10)),
            ]),
            toga.Box(style=Pack(direction=ROW, align_items=CENTER), children=[
                toga.Label("Taille du saut (hop size): ", style=Pack(text_align=CENTER, flex=0.5, margin=10)),
                toga.NumberInput(value=256, style=Pack(flex=0.5, margin=10)),
            ]),
            toga.Box(style=Pack(direction=ROW, align_items=CENTER), children=[
                toga.Label("Nombre de fréquences conservées: ", style=Pack(text_align=CENTER, flex=0.5, margin=10)),
                toga.NumberInput(value=3, style=Pack(flex=0.5, margin=10)),
            ])
        ])

        self.traiter_signal = toga.Box(style=Pack(margin=5, direction=ROW), children=[
            toga.Box(flex=0.3),
            toga.Button("Traiter le signal", on_press=self.calculate_stft, style=Pack(margin=5, flex=0.4)),
            toga.Box(flex=0.3)
        ])

        self.psnr_label = toga.Label("PSNR: N/A", style=Pack(margin=5, text_align=CENTER, visibility='hidden'))

        self.playback_box = toga.Box(style=Pack(direction=ROW, margin=5, gap=60), children=[
            toga.Button("Lire le signal original", on_press=self.play_original, style=Pack(margin=5, flex=0.5, margin_left=30)),
            toga.Button("Lire le signal reconstruit", on_press=self.play_reconstructed, style=Pack(margin=5, flex=0.5, margin_right=30))
        ])

        self.graphiques_box = toga.Box(style=Pack(direction=COLUMN), children=[
            toga.Box(style=Pack(direction=ROW, margin=5, gap=60),children=[
                toga.Button("Afficher les signaux", on_press=self.show_signals, style=Pack(margin=5, flex=0.5, margin_left=30)),
                toga.Button("Afficher les spectrogrammes", on_press=self.show_spectrograms, style=Pack(margin=5, flex=0.5, margin_right=30))
            ]),

            toga.Box(style=Pack(margin=5, direction=ROW), children=[
                toga.Box(flex=0.3),
                toga.Button("Afficher les notes", on_press=self.show_notes, style=Pack(margin=5, flex=0.4)),
                toga.Box(flex=0.3)
            ])
        ])

        self.traitement_audio.add(self.audio)
        self.traitement_audio.add(self.label_audio)
        self.traitement_audio.add(self.parametres)
        self.traitement_audio.add(self.traiter_signal)
        self.traitement_audio.add(self.psnr_label)
        self.traitement_audio.add(self.playback_box)
        self.traitement_audio.add(self.graphiques_box)

        self.main_box.add(self.traitement_audio)
        self._icon=toga.Icon(Path.absolute(Path.joinpath(paths.Paths().app,"resources","icons", "icon.png")))
        self.save_command=toga.Command.standard(
            self,
            toga.Command.SAVE, 
            action=self.save_audio, 
            icon=Path.absolute(Path.joinpath(paths.Paths().app,"resources","icons", "icon.png")),
            text="Sauvegarder l'audio modifié")
        self.load_command=toga.Command.standard(
            self,
            toga.Command.OPEN,
            action=self.load_audio,
            text="Ouvrir un fichier son"
        )
        self.about_command=toga.Command.standard(
            self,
            toga.Command.ABOUT,
            action=self.about,
            icon=Path.absolute(Path.joinpath(paths.Paths().app,"resources","icons", "icon.png"))
        )
        
        self.commands.add(self.save_command)
        self.commands.add(self.load_command)
        self.main_window.toolbar.add(self.save_command)




    async def load_audio(self, widget):
        """Charge un fichier audio WAV et en extrait le signal et la fréquence d'échantillonnage."""
        self.signal = self.sample_rate = self.signal_reconstructed = self.stft_matrix = self.stft_matrix_reconstructed = self.notes = None
        self.psnr_label.visibility = 'hidden'
        path_ini=Path.absolute(Path.joinpath(paths.Paths().app,"resources","sounds"))
        file_path = await self.main_window.dialog(
            toga.OpenFileDialog("Choisissez votre fichier audio",
                                file_types=["wav"], multiple_select=False, initial_directory=path_ini)
        )
        if file_path:
            self.file_path = file_path
            self.label_audio.text = f"Fichier chargé: {str(self.file_path).split('/')[-1]}"
            self.sample_rate, self.signal = compression_stft.load_audio_file(self.file_path)
        else:
            self.label_audio.text = "Aucun fichier sélectionné"
    
    async def save_audio(self,widget):
        """Sauvegarde le signal audio reconstruit dans un fichier WAV."""
        if self.signal_reconstructed is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal."))
            return
        path = await self.main_window.dialog(toga.SaveFileDialog("Enregistrer le fichier audio reconstruit",file_types=["wav"],suggested_filename="reconstructed.wav"))
        if path is not None:
            compression_stft.save_audio_file(self.signal_reconstructed, self.sample_rate, path)

    async def record_audio(self, widget):
        """Enregistre un audio depuis le microphone.
        Fonctionne avec sounddevice ou miniaudio selon la disponibilité.
        """
        self.signal = self.sample_rate = self.signal_reconstructed = self.stft_matrix = self.stft_matrix_reconstructed = self.notes = None
        self.psnr_label.visibility = 'hidden'

        if not self.recording:
            self.recording = True
            self.label_audio.text = "Enregistrement en cours..."
            self.audio.children[1].text = "Arrêter l'enregistrement"
            self.buffer_chunks = []

            if miniaudio_available == True:
                def record_to_buffer():
                    _ = yield
                    while True:
                        data = yield
                        self.buffer_chunks.append(data)

                self.capture = miniaudio.CaptureDevice(sample_rate=44100, nchannels=1, input_format=miniaudio.SampleFormat.FLOAT32)
                self.generator = record_to_buffer()
                next(self.generator)
                self.capture.start(self.generator)
            else:
                def callback(indata, frames, time, status):
                    self.buffer_chunks.append(indata.copy().tobytes())

                self.stream = sd.InputStream(samplerate=44100, channels=1, dtype='float32', callback=callback)
                self.stream.start()


        else:
            self.recording = False
            self.audio.children[1].text = "Enregistrer un audio"
            self.label_audio.text = "Audio enregistré"
            self.sample_rate = 44100
            if miniaudio_available == True:
                    
                self.capture.stop()

                buffer = b"".join(self.buffer_chunks)
                self.signal = np.frombuffer(buffer, dtype=np.float32)
            else:
                self.stream.stop()
                self.stream.close()
                buffer = b"".join(self.buffer_chunks)
                self.signal = np.frombuffer(buffer, dtype=np.float32).flatten()



    async def play_original(self, widget):
        """Joue le signal audio original."""
        if self.signal is not None:
            if miniaudio_available == False:
                sd.play(self.signal, self.sample_rate)
                duration = len(self.signal) / self.sample_rate
                await asyncio.sleep(duration)

            else:
                stream = memory_stream(self.signal.astype(np.float32))
                duration = len(self.signal) / self.sample_rate
                with miniaudio.PlaybackDevice(nchannels=1, sample_rate=self.sample_rate, output_format=miniaudio.SampleFormat.FLOAT32) as device:
                    next(stream)
                    device.start(stream)
                    await asyncio.sleep(duration)
        else:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Aucun fichier audio chargé ou enregistré."))


    async def play_reconstructed(self, widget):
        """Joue le signal reconstruit à partir des notes de musique."""
        if self.signal_reconstructed is not None:
            if miniaudio_available == False:
                sd.play(self.signal_reconstructed, self.sample_rate)
                duration = len(self.signal_reconstructed) / self.sample_rate
                await asyncio.sleep(duration)
            else:
                stream = memory_stream(self.signal_reconstructed.astype(np.float32))
                duration = len(self.signal_reconstructed) / self.sample_rate
                with miniaudio.PlaybackDevice(nchannels=1, sample_rate=self.sample_rate, output_format=miniaudio.SampleFormat.FLOAT32) as device:
                    next(stream)
                    device.start(stream)
                    await asyncio.sleep(duration)
        else:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal."))


    async def calculate_stft(self, widget):
        """Fonction principale de traitement du signal audio.
        Elle effectue les étapes suivantes:
        1. Vérifie que le signal audio est chargé.
        2. Récupère les paramètres de traitement (taille de la fenêtre, taille du saut, nombre de fréquences).
        3. Effectue le padding du signal pour éviter les effets de bord.
        4. Calcule la STFT du signal.
        5. Arrondit les fréquences pour obtenir des notes de musique.
        6. Reconstruit le signal audio à partir des notes.
        7. Calcule et affiche le PSNR entre le signal original et le signal reconstruit.
        """
        if self.signal is None:
            # Si aucun signal n'est chargé, on retourne une erreur
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Aucun fichier audio chargé ou enregistré."))
            return

        self.frame_size = int(self.parametres.children[0].children[1].value)
        self.hop_size = int(self.parametres.children[1].children[1].value)
        self.nombre_de_freq = int(self.parametres.children[2].children[1].value)

        if self.frame_size <= 0 or self.hop_size <= 0 or self.nombre_de_freq <= 0:
            # Si les paramètres sont invalides, on retourne une erreur
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Les paramètres doivent être positifs."))
            return
        if self.frame_size < self.hop_size:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "La taille de la fenêtre doit être supérieure ou égale à la taille du saut."))
            return
        if self.frame_size%2!=0:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "La taille de la fenêtre doit être paire."))
            return
        # Essayez de comprendre cette ligne, elle est marrante (bon courage)
        if not(self.frame_size & self.frame_size-1==0) or not(self.hop_size & self.hop_size-1==0):
            await self.main_window.dialog(toga.InfoDialog("Attention", "Il est recommandé d'utiliser des tailles de fenêtre et de saut étant des puissances de 2."))

        # Padding du signal pour éviter les effets de bord, l'effet Gibbs pour les curieux
        pad_width = self.frame_size
        data_padded = np.pad(self.signal, (pad_width, pad_width), mode="constant")

        # Voir compression_stft.py pour comprendre ces classes
        self.fft = compression_stft.FFT(self.frame_size, self.hop_size)
        self.note = compression_stft.Note()

        # On convertit le signal en fréquences temporelles, grâce à la STFT
        self.stft_matrix = self.fft.stft(data_padded)

        # On arrondit les fréquences pour obtenir des notes de musique
        self.stft_matrix_reconstructed, self.notes = self.note.justifieur(
            self.stft_matrix, self.sample_rate, self.frame_size, self.nombre_de_freq
        )

        # On reconvertit les fréquences temporelles en signal audio
        data2_padded = self.fft.istft(self.stft_matrix_reconstructed).astype(np.float32)

        #On enlève le padding
        self.signal_reconstructed = data2_padded[pad_width:-pad_width]

        # Vu qu'on a supprimé des fréquences (et leur intensité), on réadapte le volume du signal reconstruit
        if max(abs(self.signal_reconstructed))!=0:
            coeff = max(abs(self.signal)) / max(abs(self.signal_reconstructed))
            self.signal_reconstructed *= coeff

        # Calcul et affichage du PSNR, qui indique la qualité de la reconstruction  
        psnr_value = compression_stft.psnr(self.signal, self.signal_reconstructed)
        self.psnr_label.text = f"PSNR (valeur élevée = perte faible): {psnr_value:.2f} dB"
        self.psnr_label.visibility = 'visible'

        await self.main_window.dialog(toga.InfoDialog("Traitement terminé", "Le signal a été traité avec succès."))


    # --- Affichage dans Canvas ---
    async def show_signals(self, widget):
        # plot_signals est une fonction globale définie plus haut
        if self.signal is None or self.signal_reconstructed is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal"))
        else:
            plot_signals(self.signal, self.signal_reconstructed, self.sample_rate)

    async def show_spectrograms(self, widget):
        # plot_spectrograms est une fonction globale définie plus haut
        if self.stft_matrix is None or self.stft_matrix_reconstructed is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal"))
        else:
            plot_spectrograms(self.stft_matrix, self.stft_matrix_reconstructed,
                              self.sample_rate, self.frame_size, self.hop_size)

    async def show_notes(self, widget):
        if self.notes is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal"))
        else:
            # plot_notes est une fonction globale définie plus haut
            plot_notes(self.notes, self.hop_size, self.sample_rate, self.nombre_de_freq)

    async def about(self, widget=None):
        """Affiche la fenêtre 'À propos'."""
        # Création de la boîte principale
        box = toga.Box(style=Pack(direction=COLUMN, alignment=CENTER, margin=20, flex=1))

        # Logo
        logo_path = str(Path.joinpath(paths.Paths().app, "resources", "icons", "icon.png"))
        logo = toga.ImageView(toga.Image(logo_path), style=Pack(width=128, height=128, padding_bottom=15))
        box.add(logo)


        # Nom de l'app
        app_label = toga.Label(
            self.formal_name,
            style=Pack(font_size=18, font_weight="bold", text_align=CENTER, margin_bottom=5)
        )
        box.add(app_label)

        # Auteur
        author_label = toga.Label(
            "Développé par Varmule",
            style=Pack(font_size=14, text_align=CENTER, margin_bottom=10)
        )
        box.add(author_label)

        version_label = toga.Label(
            f"Version: {self.version}",
            style=Pack(font_size=12, text_align=CENTER, margin_bottom=10)
        )
        box.add(version_label)

        # Lien cliquable
        def open_link(widget):
            webbrowser.open("https://github.com/varmule/sound2note")

        link = toga.Button(
            "Code source du project",
            on_press=open_link,
            style=Pack(margin_top=5)
        )
        box.add(link)

        # Boîte de dialogue
        dialog = toga.Window(title="À propos de Sound2note")
        dialog.content = box
        dialog.show()






def main():
    return Sound2note()

if __name__=="__main__":
    main().main_loop()