import os
import pickle
import numpy as np
import librosa
from keras.models import load_model
from utils.audio import cqhc

def emotion_classification(wave_file, upload_folder):
    data, sr = librosa.load(wave_file)
    # Input shape - (20,290,1)
    feat = cqhc(data, sr, min_freq=30, octave_resolution=14, num_coeff=20)
    
    # Pad = 290
    shape = 290 - feat.shape[1]
    if shape < 0:
        feat = feat[:, :290]
        shape = 0
    feat_pad = np.pad(feat, ((0,0), (0,shape)), 'constant')
    feat_pad = feat_pad.reshape(1, 20, 290, 1)
    
    model = load_model('models/weights/emotion_h5_file.h5')
    ans = model(feat_pad)
    output = np.argmax(ans)
    
    labels_map = {
        0: ('angry.gif', 'Angry'),
        1: ('happy.gif', 'Happy'),
        2: ('neutral.gif', 'Neutral'),
        3: ('sad.gif', 'Sad'),
        4: ('surprise.gif', 'Surprise')
    }
    
    img_name, label_text = labels_map.get(output, ('neutral.gif', 'Neutral'))
    img_path = os.path.join(upload_folder, img_name).replace('\\', '/')
    return img_path, label_text
