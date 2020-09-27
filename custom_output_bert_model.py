import os
from transformers import BertConfig, BertForPreTraining, BertTokenizer


CUSTOM_BERT_CONFIG_DICT = {
    "bert_mini": {
        "output_hidden_states": True,
        "config_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_model_dir/bert_mini/bert_config.json"),
        "index_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_model_dir/bert_mini/bert_model.ckpt.index")
    },
    "bert_base": {
        "output_hidden_states": True,
        "config_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_model_dir/bert_base/bert_config.json"),
        "index_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_model_dir/bert_base/bert_model.ckpt.index")
    },
}

class FeatureExtractorBert(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, **inputs):
        outputs = super().forward(**inputs)
        return outputs[2][0]

    def model_builder(model_name):
        assert model_name in CUSTOM_BERT_CONFIG_DICT
        config = BertConfig.from_json_file(CUSTOM_BERT_CONFIG_DICT[model_name]["config_file"])
        config.update({"output_hidden_states": True})
        return FeatureExtractorBert.from_pretrained(CUSTOM_BERT_CONFIG_DICT[model_name]["index_file"], config=config, from_tf=True)
    
