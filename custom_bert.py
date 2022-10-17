from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch
from typing import Optional
from bert_processing import MAX_LEN

class CustomBert(BertPreTrainedModel):
    """
    BertPreTrainedModel with a regression layer on top. Supports adding additional features to the BERT output.
    """
    def __init__(self, config, num_custom_features):
        super().__init__(config)
        self.config = config
        self.config.problem_type = "regression"

        self.bert = BertModel(config)
        self.num_custom_features = num_custom_features
        self.predictor = torch.nn.Linear(config.hidden_size + num_custom_features, 1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None):
        
        # Cut away our custom features.
        bert_input = input_ids[:, :-self.num_custom_features]
        output = self.bert.forward(bert_input, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, False)
        pooled_output = output[1]

        # Append custom features to BERT output, and feed into prediction layer.
        predictor_input = torch.cat((pooled_output, input_ids[:, -self.num_custom_features:]), dim=1)
        pred_out = self.predictor(predictor_input)

        if labels is not None:
            loss_function = torch.nn.MSELoss()
            return loss_function(pred_out.squeeze(), labels.squeeze()), pred_out
        else:
            return pred_out