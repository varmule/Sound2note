from compression_stft import *
from tkinter import *
from tkinter import filedialog, messagebox, ttk, scrolledtext
import sounddevice as sd
import sys
import webbrowser

original_stdout = sys.stdout # Save a reference to the original standard output
original_stderr = sys.stderr # Save a reference to the original standard error

class RedirectText:
    """Redirige stdout/stderr vers un widget Text sans bloquer l'UI"""
    def __init__(self, text_widget, max_lines=1000, flush_interval=50):
        self.output = text_widget # Widget Text
        self.max_lines = max_lines # Nombre maximum de lignes à conserver
        self.flush_interval = flush_interval  # Intervalle de temps pour vider le buffer (en ms)
        self.buffer = [] # Buffer pour stocker les messages
        self.scheduled = False 
    def write(self, string):
        self.buffer.append(string) # Ajouter le message au buffer
        if not self.scheduled: 
            self.output.after(self.flush_interval, self.flush_buffer) # Vide le buffer après un intervalle
            self.scheduled = True
    def flush(self): # Nécessaire, mais pas utilisé
        pass
    def flush_buffer(self):
        if self.buffer:
            text = "".join(self.buffer)
            self.output.insert("end", text)
            self.output.see("end")
            self.buffer.clear()
            line_count = int(self.output.index("end-1c").split(".")[0])
            if line_count > self.max_lines:
                self.output.delete("1.0", f"{line_count - self.max_lines}.0")
        self.scheduled = False



class GUI(object):
    def __init__(self, master):
        """Interface graphique pour Sound2Note"""

        self.master = master # Fenêtre principale
        self.master.title("Sound2Note GUI")
        self.master.resizable(False, False)
        # Variables pour le signal et la STFT
        self.file_path = None
        self.recording = False
        self.signal = None
        self.sample_rate = None
        self.signal_reconstructed = None
        self.siglal_stft_matrix = None
        self.stft_matrix_reconstructed = None

        # Création de l'interface
        self.setup_ui()
    def setup_ui(self):
        self.notebook=ttk.Notebook(self.master) # Onglets

        self.traitement = ttk.Frame(self.notebook) # Onglet Traitement audio
        self.traitement.grid_columnconfigure(0, weight=1)
        self.traitement.grid_columnconfigure(1, weight=1)
        for i in range(8):
            self.traitement.grid_rowconfigure(i, weight=1)
        
        self.terminal = ttk.Frame(self.notebook) # Onglet Sortie du terminal
        self.terminal.grid_rowconfigure(0, weight=1)
        self.terminal.grid_columnconfigure(0, weight=1)

        self.about = ttk.Frame(self.notebook) # Onglet A propos
        self.about.grid_rowconfigure(0, weight=1)
        self.about.grid_columnconfigure(0, weight=1)

        # Ajout des onglets au notebook
        self.notebook.add(self.traitement, text="Traitement audio")
        self.notebook.add(self.terminal, text="Sortie du terminal")
        self.notebook.add(self.about, text="A propos")

        # Bouton pour charger un fichier audio
        self.load_button = ttk.Button(self.traitement, text="Charger un fichier audio", command=self.load_audio)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)
        # Bouton pour enregistrer un audio via le micro
        self.record_button = ttk.Button(self.traitement, text="Enregistrer un audio", command=self.record)
        self.record_button.grid(row=0, column=1, padx=10, pady=10)
        # Label pour afficher le chemin du fichier chargé
        self.file_label = ttk.Label(self.traitement, text="Aucun fichier sélectionné")
        self.file_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Entrées pour les paramètres de la STFT : frame size, hop size, nombre de fréquences
        self.frame_label = ttk.Label(self.traitement, text="Taille de la fenêtre (frame size):")
        self.frame_label.grid(row=2, column=0, padx=10, pady=10)
        self.frame_entry = ttk.Entry(self.traitement)
        self.frame_entry.insert(0, "1024")
        self.frame_entry.grid(row=2, column=1, padx=10, pady=10)
        self.hop_label = ttk.Label(self.traitement, text="Taille du saut (hop size):")
        self.hop_label.grid(row=3, column=0, padx=10, pady=10)
        self.hop_entry = ttk.Entry(self.traitement)
        self.hop_entry.insert(0, "256")
        self.hop_entry.grid(row=3, column=1, padx=10, pady=10)
        self.freq_label = ttk.Label(self.traitement, text="Nombre de fréquences conservées:")
        self.freq_label.grid(row=4, column=0, padx=10, pady=10)
        self.freq_entry = ttk.Entry(self.traitement)
        self.freq_entry.insert(0, "3")
        self.freq_entry.grid(row=4, column=1, padx=10, pady=10)

        # Bouton principal : modifie le signal en arrondissant les fréquences les plus importantes aux notes les plus proches
        self.calculate_button = ttk.Button(self.traitement, text="Traiter le signal", command=self.calculate_stft)
        self.calculate_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # Label pour afficher le PSNR
        self.psnr_label = ttk.Label(self.traitement, text="PSNR (valeur élevée = perte faible): N/A")
        self.psnr_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        # Boutons pour jouer les signaux et afficher les graphiques
        self.play_original_button = ttk.Button(self.traitement, text="Jouer le signal original", command=self.play_original)
        self.play_original_button.grid(row=7, column=0, padx=10, pady=10)
        self.play_reconstructed_button = ttk.Button(self.traitement, text="Jouer le signal reconstruit", command=self.play_reconstructed)
        self.play_reconstructed_button.grid(row=7, column=1, padx=10, pady=10)

        self.show_signals_button = ttk.Button(self.traitement, text="Afficher les signaux", command=self.show_signals)
        self.show_signals_button.grid(row=8, column=0, padx=10, pady=10)
        self.show_spectrogram_button = ttk.Button(self.traitement, text="Afficher les spectrogrammes", command=self.show_spectrograms)
        self.show_spectrogram_button.grid(row=8, column=1, padx=10, pady=10)

        # Bouton pour sauvegarder le signal reconstruit
        self.save_button = ttk.Button(self.traitement, text="Sauvegarder le signal reconstruit", command=self.save_reconstructed)
        self.save_button.grid(row=9, column=0, columnspan=2, padx=10, pady=10)


        # Terminal (onglet)
        self.output_terminal = scrolledtext.ScrolledText(self.terminal, wrap=WORD)
        self.output_terminal.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        # Ces deux lignes redirigent stdout et stderr (la sortie du terminal) vers le widget Text
        sys.stdout = RedirectText(self.output_terminal, max_lines=1000, flush_interval=50)
        sys.stderr = RedirectText(self.output_terminal, max_lines=1000, flush_interval=50)



        center_frame = ttk.Frame(self.about)
        center_frame.grid(row=0, column=0, sticky="nsew")
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)

        # Frame pour les labels (contenu réel)
        content_frame = ttk.Frame(center_frame)
        content_frame.grid(row=0, column=0, sticky="n", pady=50)  # pady pour centrer verticalement

        # Titre
        title_label = ttk.Label(content_frame, text="Sound2Note GUI", font=("Helvetica", 32, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # Version
        version_label = ttk.Label(content_frame, text="Version 2.0", font=("Helvetica", 20))
        version_label.grid(row=1, column=0, pady=(0, 5))

        # Description
        desc_label = ttk.Label(content_frame,text="Application pour charger, traiter et reconstruire des signaux audio\nà partir de la STFT.",font=("Helvetica", 20),justify="center")
        desc_label.grid(row=2, column=0, pady=(0, 5))

        # Auteur
        author_label = ttk.Label(content_frame, text="Développé par Varmule", font=("Helvetica", 18, "italic"))
        author_label.grid(row=3, column=0, pady=(0, 5))

        # Lien GitHub
        link_label = ttk.Label(content_frame, text="Code source disponible sur GitHub",cursor="pointinghand", foreground="blue", font=("Helvetica", 16, "underline"))
        link_label.grid(row=4, column=0, pady=(10, 0))
        link_label.bind("<Button-1>", self.open_website)

        self.notebook.pack(expand=1, fill="both")
    def load_audio(self):
        """Charge un fichier audio et affiche son chemin"""
        self.file_path=None
        self.signal=None
        self.sample_rate=None
        self.signal_reconstructed=None
        self.stft_matrix_reconstructed=None
        self.stft_matrix=None
        self.psnr_label.config(text="PSNR (valeur élevée = perte faible): N/A")
        file_path = filedialog.askopenfilename(filetypes=[("Fichiers WAV", "*.wav")], initialdir="sounds")
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Fichier chargé: {self.file_path.split('/')[-1]}")
            self.sample_rate, self.signal = load_audio_file(self.file_path)

        else:
            self.file_label.config(text="Aucun fichier sélectionné")
    def record(self):
        """Enregistre un audio via le micro"""
        if not self.recording:
            self.recording = True
            self.record_button.config(text="Arrêter l'enregistrement")
            def callback(indata, frames, time, status):
                if self.recording:
                    self.signal = np.append(self.signal, indata.copy())
            self.signal = np.empty((0, 1), dtype=np.float32)
            self.sample_rate = 44100  # Fréquence d'échantillonnage standard
            self.file_label.config(text="Enregistrement en cours...")
            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback)
            self.stream.start()
        else:
            self.recording = False
            self.record_button.config(text="Enregistrer un audio")
            self.stream.stop()
            self.stream.close()
            self.signal = self.signal.flatten()
            self.file_label.config(text="Audio enregistré via le micro")
    def calculate_stft(self):
        """Calcule la STFT et la reconstruction du signal"""
        if self.signal is None:
            messagebox.showerror("Erreur", "Veuillez d'abord charger un fichier audio.")
            return
        try:
            frame_size = int(eval(self.frame_entry.get()))
            hop_size = int(eval(self.hop_entry.get()))
            nombre_de_freq = int(eval(self.freq_entry.get()))
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs entières valides pour la taille de la fenêtre, le saut et le nombre de fréquences.")
            return
        pad_width = frame_size  # 1 frame de chaque côté
        data_padded = np.pad(self.signal, (pad_width, pad_width), mode="constant")
        fft=FFT(frame_size,hop_size)
        note=Note()
        self.stft_matrix=fft.stft(data_padded)
        self.stft_matrix_reconstructed=note.justifieur(self.stft_matrix,self.sample_rate,frame_size,nombre_de_freq)
        data2_padded=fft.istft(self.stft_matrix_reconstructed).astype(np.float32)

        self.signal_reconstructed = data2_padded[pad_width:-pad_width]

        coeff=max(abs(self.signal))/max(abs(self.signal_reconstructed))
        self.signal_reconstructed*=coeff
        psnr_value = psnr(self.signal, self.signal_reconstructed)
        self.psnr_label.config(text=f"PSNR (valeur élevée = perte faible): {psnr_value:.2f} dB")
        messagebox.showinfo("Succès", "STFT et reconstruction terminées.")
        self.save_button.config(state=NORMAL)
    def play_original(self):
        """Joue le signal audio original"""
        if self.signal is None:
            messagebox.showerror("Erreur", "Aucun signal chargé à jouer.")
            return
        sd.play(self.signal, self.sample_rate)
        sd.wait()
    def play_reconstructed(self):
        """Joue le signal audio reconstruit"""
        if self.signal_reconstructed is None:
            messagebox.showerror("Erreur", "Aucun signal reconstruit à jouer.")
            return
        sd.play(self.signal_reconstructed, self.sample_rate)
        sd.wait()
    def show_signals(self):
        """Affiche les signaux original et reconstruit"""
        if self.signal is None or self.signal_reconstructed is None:
            messagebox.showerror("Erreur", "Les signaux ne sont pas prêts à être affichés.")
            return
        t1 = np.linspace(0, len(self.signal) / self.sample_rate, len(self.signal))
        t2 = np.linspace(0, len(self.signal_reconstructed) / self.sample_rate, len(self.signal_reconstructed))
        Graphiques().signaux(self.signal, self.signal_reconstructed, t1, t2)
        plt.show()
    def show_spectrograms(self):
        """Affiche les spectrogrammes original et reconstruit"""
        if self.stft_matrix is None or self.stft_matrix_reconstructed is None:
            messagebox.showerror("Erreur", "Les spectrogrammes ne sont pas prêts à être affichés.")
            return
        t1 = np.linspace(0, len(self.signal) / self.sample_rate, len(self.signal))
        t2 = np.linspace(0, len(self.signal_reconstructed) / self.sample_rate, len(self.signal_reconstructed))
        Graphiques().spectrogrammes(self.stft_matrix, self.stft_matrix_reconstructed, t1, t2)
        plt.show()
    def save_reconstructed(self):
        """Sauvegarde le signal reconstruit dans un fichier WAV"""
        if self.signal_reconstructed is None:
            messagebox.showerror("Erreur", "Aucun signal reconstruit à sauvegarder.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Fichiers WAV", "*.wav")], initialfile="reconstructed.wav", initialdir="sounds")
        if save_path:
            save_audio_file(save_path, self.sample_rate, self.signal_reconstructed)
            messagebox.showinfo("Succès", f"Signal reconstruit sauvegardé sous: {save_path}")
        # Bouton pour sauvegarder le signal reconstruit
    def open_website(self, event):
        webbrowser.open_new("https://github.com/varmule/Sound2Note")
if __name__ == "__main__":
    root = Tk()
    app = GUI(root)
    try:
        root.mainloop()
    finally:
        # Restaurer stdout/stderr pour éviter les problèmes
        sys.stdout = original_stdout
        sys.stderr = original_stderr