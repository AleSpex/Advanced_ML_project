import torch
import torch.nn as nn



from transformers import BertTokenizer, BertModel

def make_encoder():
    return BertEncoder()

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.out_dim = 768
        self.eval()

    def forward(self, sentences, device='cuda'):
        """
        sentences: list[str], len of list: B
        output: embeddings
        """
        embeddings = list()
        for sentence in sentences:
            text = '[CLS] ' + sentence + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(indexed_tokens)
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            segments_tensors = torch.tensor([segments_ids]).to(device)

            # TODO
            # I have to change this:
            #last_hidden_states, _ = self.model(tokens_tensor, segments_tensors)
            #sentence_embedding = torch.mean(last_hidden_states[0], dim=0)

            #in this:
            outputs = self.model(tokens_tensor, segments_tensors)
            last_hidden_states = outputs.last_hidden_state
            sentence_embedding = torch.mean(last_hidden_states[0], dim=0)

            # del tokens_tensor, segments_tensors, last_hidden_states
            embeddings.append(sentence_embedding)
        embeddings = torch.stack(embeddings)
        return embeddings
