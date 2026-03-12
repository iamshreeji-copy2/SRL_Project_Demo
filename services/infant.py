import os
import pickle
import numpy as np
import librosa

def infant_cry_classification(sound_file, upload_folder):
    audio, sr = librosa.load(sound_file)
    
    # Check for no crying voice or silence (low energy level)
    rms = np.mean(librosa.feature.rms(y=audio))
    if rms < 0.01:
        img_path = os.path.join(upload_folder, 'blank.png').replace('\\', '/')
        return img_path, 'No Crying'

    # Heuristic to distinguish normal voice (non-crying) from infant crying
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    # Extract dominant pitches
    mask = magnitudes > np.max(magnitudes) * 0.5
    if np.any(mask):
        pitch_vals = pitches[mask]
        avg_pitch = np.mean(pitch_vals)
        if avg_pitch < 300: # Threshold for adult speech/normal voice
            img_path = os.path.join(upload_folder, 'blank.png').replace('\\', '/')
            return img_path, 'No Crying'

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    test_extracted = mfccs_processed.reshape((1, -1))
    
    classifier_mod = pickle.load(open('models/weights/knnpickle_file', 'rb'))
    y_pred = classifier_mod.predict(test_extracted)
    
    if y_pred == 0:
        img_name, label_text = 'normal.png', 'Normal cry'
    elif y_pred == 1:
        img_name, label_text = 'pathological.png', 'Pathology cry'
    else:
        img_name, label_text = 'blank.png', 'No Crying'
        
    img_path = os.path.join(upload_folder, img_name).replace('\\', '/')
    return img_path, label_text
