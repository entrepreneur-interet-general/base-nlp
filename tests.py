from transformers import CamembertConfig, CamembertModel, CamembertTokenizer, BertTokenizer, BertModel, BertConfig
import torch
from torch.ao.ns.fx.utils import compute_cosine_similarity

transformer_version = "bert-base-multilingual-cased"

if "camembert" in transformer_version:
    tokenizer = CamembertTokenizer.from_pretrained(transformer_version)
    config = CamembertConfig.from_pretrained(transformer_version)
    camembert = CamembertModel.from_pretrained(transformer_version, config=config)
else:
    tokenizer = BertTokenizer.from_pretrained(transformer_version)
    config = BertConfig.from_pretrained(transformer_version)
    camembert = BertModel.from_pretrained(transformer_version, config=config)


base_sentence = """
cerises
"""

compared_sentences = [
    "cherries",
    "fraises", 
    "kirschen",
    ]


def encode_sentence(source_sentence):
    encoded_sentence = tokenizer.encode(source_sentence)
    encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)

    return encoded_sentence


def get_embeddings(source_sentence):
    encoded_sentence = encode_sentence(source_sentence)
    embeddings = camembert(encoded_sentence)['last_hidden_state']

    return embeddings


def get_mean_embeddings(source_sentence):
    embeddings = get_embeddings(source_sentence)
    #  all_layer_embeddings list of len(all_layer_embeddings) == 13 (input embedding layer + 12 self attention layers)
    pipou = torch.mean(embeddings, dim=1)
    return pipou


# TODO faire marcher le truc
def get_peak_embeddings(source_sentence):
    embeddings = get_embeddings(source_sentence)

    abs_embeddings = torch.abs(embeddings)
    max_abs_embeddings = torch.max(abs_embeddings, dim=1).indices

    pipou = embeddings[max_abs_embeddings]

    return pipou


base_embedding = get_mean_embeddings(base_sentence)
compared_embeddings = [get_mean_embeddings(sentence) for sentence in compared_sentences]

for ix, sentence in enumerate(compared_sentences):
    print(sentence, compute_cosine_similarity(base_embedding, compared_embeddings[ix]))