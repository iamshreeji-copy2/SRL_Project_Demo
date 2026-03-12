import os
import io
import pickle
import torch
import librosa
from models.resnet import ResNet, BasicBlock, resnet50

class ModelUnpickler(pickle.Unpickler):
    """Custom unpickler to handle class references that were pickled under __main__."""
    def find_class(self, module, name):
        # Remap classes that were pickled under __main__ to their actual module
        if name == 'ResNet':
            return ResNet
        if name == 'BasicBlock':
            return BasicBlock
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def audio_deepfake_detection(sound_file, upload_folder, processor, model):
    sample, sr = librosa.load(sound_file, sr=16000)
    input_feature = processor(sample, sampling_rate=sr, return_tensors="pt").input_features
    
    with torch.no_grad():
        images = model(input_feature).last_hidden_state
    
    with open('models/weights/model.pkl', 'rb') as f:
        automl = ModelUnpickler(f).load()
    automl.eval()
    
    images = images[-1, 0:100, :]
    images = images.unsqueeze(0).unsqueeze(0).float()
    
    outputs = automl(images)
    _, prediction = torch.max(outputs.data, 1)
    
    y_pred = prediction.float()
    if y_pred == 1:
        img_name, label_text = '123.png', 'Real'
    else:
        img_name, label_text = 'jh.png', 'Fake'
        
    img_path = os.path.join(upload_folder, img_name).replace('\\', '/')
    return img_path, label_text
