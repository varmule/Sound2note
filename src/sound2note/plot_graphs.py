# src/sound2note/plot_worker.py
"""
Module exécuté dans un process séparé pour afficher les graphiques.
Usage :
  python -m sound2note.plot_worker show_signals signal.npy recon.npy sample_rate
  python -m sound2note.plot_worker show_spectrograms stft.npy stft_recon.npy sample_rate frame_size hop_size
  python -m sound2note.plot_worker show_notes notes.npy hop_size sample_rate n_freq
"""

import sys
import argparse
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")  # autorisé ici, car process séparé
from matplotlib import pyplot as plt
from sound2note import compression_stft

def show_signals(args):
    sig = np.load(args.signal)
    rec = np.load(args.reconstructed)
    sr = float(args.sample_rate)
    t1 = np.arange(len(sig)) / sr
    t2 = np.arange(len(rec)) / sr
    compression_stft.Graphiques().signaux(sig, rec, t1, t2)
    plt.show()

def show_spectrograms(args):
    stft1 = np.load(args.stft)
    stft2 = np.load(args.reconstructed)
    sr = float(args.sample_rate)
    fs = int(args.frame_size)
    hop = int(args.hop_size)
    compression_stft.Graphiques().spectrogrammes(stft1, stft2, sr, fs, hop)
    plt.show()

def show_notes(args):
    notes = np.load(args.notes, allow_pickle=True).tolist()
    hop = int(args.hop_size)
    sr = float(args.sample_rate)
    nf = int(args.n_freq)
    compression_stft.Graphiques().piano_roll(notes, hop, sr, nf)
    plt.show()

def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("show_signals")
    p1.add_argument("signal")
    p1.add_argument("reconstructed")
    p1.add_argument("sample_rate")

    p2 = sub.add_parser("show_spectrograms")
    p2.add_argument("stft")
    p2.add_argument("reconstructed")
    p2.add_argument("sample_rate")
    p2.add_argument("frame_size")
    p2.add_argument("hop_size")

    p3 = sub.add_parser("show_notes")
    p3.add_argument("notes")
    p3.add_argument("hop_size")
    p3.add_argument("sample_rate")
    p3.add_argument("n_freq")

    args = parser.parse_args(argv)
    if args.cmd == "show_signals":
        show_signals(args)
    elif args.cmd == "show_spectrograms":
        show_spectrograms(args)
    elif args.cmd == "show_notes":
        show_notes(args)

if __name__ == "__main__":
    main()
