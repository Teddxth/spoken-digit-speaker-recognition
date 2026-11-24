# Spoken Digit & Speaker Recognition: CNN on audio spectrograms

Two classification tasks on the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) (6 speakers × 10 digits × 50 recordings):

1. **Which digit is spoken?** (10 classes)
2. **Who is speaking?** (6 classes)

It also includes a **multi-input fusion network** that combines a mel-spectrogram CNN with an FFT-spectrum network to improve digit recognition.

> Projet de 1re année d'ingénieur (Arts et Métiers, PJT1A). Le notebook de départ (chargement des données, fonctions utilitaires) était fourni par l'école ; les architectures, l'étude FFT et le réseau de fusion sont mon travail.

---

## Pipeline

```
audio .wav ──► mel-spectrogram (128 × 36) ──► CNN ─────────────┐
     └──────► FFT magnitude spectrum ─────► dense network ────┼──► fusion network ──► digit (0-9)
                                                              │
mel-spectrogram ──► CNN ──► speaker (6 classes)               ┘
```

- **Pre-processing:** librosa mel-spectrograms (128 mel bands), zero-padding to a fixed length, cached as `.npy`
- **Split:** per speaker *and* per digit, so every class is equally represented in train and test (1 200 test samples)
- **Models (Keras):** 2 × Conv2D + dense head for spectrograms; dense network on FFT amplitudes; a small fusion network trained on the concatenated predictions of both

## Results (test set, 1 200 samples)

| Model | Task | Accuracy |
|---|---|---|
| CNN on mel-spectrogram | Speaker (6 classes) | **91.7 %** |
| CNN on mel-spectrogram | Digit (10 classes) | 80.1 % |
| Dense network on FFT spectrum | Digit | 78.3 % |
| **Fusion (mel CNN + FFT net)** | Digit | **83.8 %** (+3.7 pts) |

**Take-aways**
- The two representations make *different* mistakes (e.g. the mel CNN confuses 1 ↔ 5 and 2 ↔ 3, the FFT net confuses 6 ↔ 8), which is exactly why fusing them helps.
- Speaker identification is easier than digit recognition: timbre is visible across the whole spectrogram, while digits differ by short phonetic details.

## Run it

Open `notebooks/spoken_digit_speaker_recognition.ipynb` in Google Colab, set your Drive folder in the first cells, and run all.

## Next steps
Data augmentation (noise, pitch shift), dropout / early stopping, end-to-end training of the fusion model instead of stacking frozen predictions.

## Author
**Teddy Thamrong**, élève-ingénieur à l'École nationale supérieure d'Arts et Métiers (ENSAM), expertise IRIS (machine learning et robotique) · recherche un stage de fin d'études en IA à partir de février ou de septembre 2027.

License: MIT
