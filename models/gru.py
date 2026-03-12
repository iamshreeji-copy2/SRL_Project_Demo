import torch
import torch.nn as nn

class BiGRUAudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_units, num_layers):
        super(BiGRUAudioClassifier, self).__init__()
        self.input_size = input_size 
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.255)
        self.fc = nn.Linear(hidden_units * 2, num_classes)

    def forward(self, x):
        # Pass the input through the bi-GRU layers
        output, _ = self.bigru(x)
        output = self.dropout(output)
        # Extract the last hidden state (concatenate forward and backward hidden states)
        last_hidden_state = torch.cat((output[:, -1, :self.hidden_units], output[:, 0, self.hidden_units:]), dim=1)
        # Apply the fully connected layer for classification
        output = self.fc(last_hidden_state)
        return output
