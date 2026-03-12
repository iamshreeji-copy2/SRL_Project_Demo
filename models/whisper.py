from typing import Optional
import torch
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperEncoder

class WhisperWordClassifier(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.post_init()

    def forward(
            self,
            input_features: Optional[torch.LongTensor] = None,  
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
        encoder_outputs = self.encoder(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
