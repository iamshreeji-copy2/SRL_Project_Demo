import os
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
from utils.audio import lfcc_mine

def voice_liveness_detection(sound_file, upload_folder):
    audio, sr = librosa.load(sound_file)
    feat = lfcc_mine(sr, audio)
    feat = feat.T
    
    train_extracted_1 = np.zeros((1, 20, 40))
    # Note: Simplified original logic which seemed to only take the last frame's padded version if multiple frames existed
    shape = 40 - feat.shape[1]
    if shape < 0:
        feat_fixed = feat[:, :40]
    else:
        feat_fixed = np.pad(feat, ((0,0), (0,shape)), 'constant')
    
    train_extracted_1[0, :, :] = feat_fixed[:, :40]
    feat_pad = train_extracted_1.reshape(1, 20, 40, 1)
    
    model = load_model('models/weights/VLD_LFCC.h5', compile=False)
    ans = model(feat_pad)
    ans = ans[0][0]
    
    y_pred = 1 if ans > 0.5 else 0
    if y_pred == 0:
        label_text = 'Genuine Voice'
        img_name = 'v1.png'
    else:
        label_text = 'Spoofed Voice'
        img_name = 'ads.png'
        
    img_path = os.path.join(upload_folder, img_name).replace('\\', '/')
    return img_path, label_text
