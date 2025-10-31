"""
App which convert a sound to notes
"""

import toga
from toga import paths
from sound2note import compression_stft
from toga.style.pack import COLUMN, ROW, CENTER, Pack
import numpy as np
import asyncio
from matplotlib import pyplot as plt
import multiprocessing
from pathlib import Path

try:
    import miniaudio
    miniaudio_available = True
except ImportError:
    import sounddevice as sd
    miniaudio_available = False


#Fonctions globales de tracé (nécessaires pour multiprocessing)


def plot_signals(signal, signal_reconstructed, sample_rate):
    t1 = np.arange(len(signal)) / sample_rate
    t2 = np.arange(len(signal_reconstructed)) / sample_rate
    compression_stft.Graphiques().signaux(signal, signal_reconstructed, t1, t2)
    plt.show()


def plot_spectrograms(stft_matrix, stft_matrix_reconstructed, sample_rate, frame_size, hop_size):
    compression_stft.Graphiques().spectrogrammes(stft_matrix, stft_matrix_reconstructed, sample_rate, frame_size, hop_size)
    plt.show()


def plot_notes(notes, hop_size, sample_rate, nombre_de_freq):
    compression_stft.Graphiques().piano_roll(notes, hop_size, sample_rate, nombre_de_freq)
    plt.show()


#Fonction audio mémoire qui sert à jouer les signaux
if miniaudio_available == True:
    def memory_stream(npa: np.ndarray) -> miniaudio.PlaybackCallbackGeneratorType:
        required_frames = yield b""  # initialise le générateur pour miniaudio
        frames = 0

        while frames < len(npa):
            frames_end = frames + required_frames
            required_frames = yield npa[frames:frames_end]
            frames = frames_end



class Sound2note(toga.App):
    """Application Toga pour convertir un son en notes musicales."""
    def startup(self):
        """Construct and show the Toga application."""
        self.main_box = toga.Box()

        # Variables pour le signal et la STFT
        self.frame_size = 1024
        self.hop_size = 256
        self.nombre_de_freq = 3

        self.file_path = None
        self.recording = False
        self.signal = None
        self.sample_rate = None
        self.signal_reconstructed = None
        self.stft_matrix = None
        self.stft_matrix_reconstructed = None
        self.notes = None

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = self.main_box
        self.setup_ui()
        self.main_window.show()


    def setup_ui(self):
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
                toga.Button("Afficher les spectrogrammes", on_press=self.show_spectograms, style=Pack(margin=5, flex=0.5, margin_right=30))
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
        self.sauvegarde=toga.Command(self.save_audio, text="Sauvegarder l'audio modifié", shortcut=toga.Key.MOD_1+'s',icon=Path.absolute(Path.joinpath(paths.Paths().app,"resources","icons", "icon.png")))
        self.commands.add(self.sauvegarde)
        self.main_window.toolbar.add(self.sauvegarde)




    async def load_audio(self, widget):
        self.signal = self.sample_rate = self.signal_reconstructed = self.stft_matrix = self.stft_matrix_reconstructed = self.notes = None
        self.psnr_label.visibility = 'hidden'
        path_ini=Path.absolute(Path.joinpath(paths.Paths().app,"resources","sounds"))
        print(path_ini)
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
        if self.signal_reconstructed is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal."))
            return
        await self.main_window.dialog(toga.SaveFileDialog("Enregistrer le fichier audio reconstruit",file_types=["wav"],suggested_filename="reconstructed.wav"))

    async def record_audio(self, widget):
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
        if self.signal is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Aucun fichier audio chargé ou enregistré."))
            return

        self.frame_size = int(self.parametres.children[0].children[1].value)
        self.hop_size = int(self.parametres.children[1].children[1].value)
        self.nombre_de_freq = int(self.parametres.children[2].children[1].value)

        if self.frame_size <= 0 or self.hop_size <= 0 or self.nombre_de_freq <= 0:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Les paramètres doivent être positifs."))
            return

        pad_width = self.frame_size
        data_padded = np.pad(self.signal, (pad_width, pad_width), mode="constant")
        self.fft = compression_stft.FFT(self.frame_size, self.hop_size)
        self.note = compression_stft.Note()

        self.stft_matrix = self.fft.stft(data_padded)
        self.stft_matrix_reconstructed, self.notes = self.note.justifieur(
            self.stft_matrix, self.sample_rate, self.frame_size, self.nombre_de_freq
        )

        data2_padded = self.fft.istft(self.stft_matrix_reconstructed).astype(np.float32)
        self.signal_reconstructed = data2_padded[pad_width:-pad_width]
        coeff = max(abs(self.signal)) / max(abs(self.signal_reconstructed))
        self.signal_reconstructed *= coeff

        psnr_value = compression_stft.psnr(self.signal, self.signal_reconstructed)
        self.psnr_label.text = f"PSNR (valeur élevée = perte faible): {psnr_value:.2f} dB"
        self.psnr_label.visibility = 'visible'

        await self.main_window.dialog(toga.InfoDialog("Traitement terminé", "Le signal a été traité avec succès."))



    async def show_signals(self, widget):
        if self.signal is None or self.signal_reconstructed is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal."))
            return
        multiprocessing.Process(target=plot_signals, args=(self.signal, self.signal_reconstructed, self.sample_rate)).start()

    async def show_spectograms(self, widget):
        if self.stft_matrix is None or self.stft_matrix_reconstructed is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal."))
            return
        multiprocessing.Process(
            target=plot_spectrograms,
            args=(self.stft_matrix, self.stft_matrix_reconstructed, self.sample_rate, self.frame_size, self.hop_size)
        ).start()

    async def show_notes(self, widget):
        if self.notes is None:
            await self.main_window.dialog(toga.ErrorDialog("Erreur", "Veuillez traiter le signal."))
            return
        multiprocessing.Process(
            target=plot_notes,
            args=(self.notes, self.hop_size, self.sample_rate, self.nombre_de_freq)
        ).start()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    return Sound2note()
