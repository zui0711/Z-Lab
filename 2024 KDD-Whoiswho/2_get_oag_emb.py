import os
from cogdl.oag import oagbert
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

tokenizer, model = oagbert("oagbert-v2-sim")
model.to('cuda')

article_info = pd.read_pickle('usr_data/article_info.pkl')
article_info['authors_info'] = article_info['authors'].map(lambda x: ' '.join([xx['name'].lower() + ' ' + xx['org'].lower() for xx in x]))

article_info['authors_name'] = article_info['authors'].map(lambda x: [xx['name'] for xx in x])
article_info['affiliations'] = article_info['authors'].map(lambda x: [ xx['org'] for xx in x])


class MyDataset(Dataset):
    def __init__(self, article_info, model):
        #         self.article_info = article_info
        self.title = article_info['title'].to_list()
        self.abstract = article_info['abstract'].to_list()
        self.venue = article_info['venue'].fillna('NULL').to_list()
        self.authors = article_info['authors_name'].to_list()
        self.concepts = article_info['keywords'].to_list()
        self.affiliations = article_info['affiliations'].to_list()

        self.model = model

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = self.model.build_inputs(
            title=self.title[idx],
            abstract=self.abstract[idx],
            venue=self.venue[idx],
            authors=self.authors[idx],
            concepts=self.concepts[idx],
            affiliations=self.affiliations[idx]
        )

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(position_ids),
            torch.LongTensor(position_ids_second)
        )


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)
    position_ids = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True, padding_value=0)
    position_ids_second = torch.nn.utils.rnn.pad_sequence([x[4] for x in batch], batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids.to('cuda'),
        'token_type_ids': token_type_ids.to('cuda'),
        'attention_mask': attention_mask.to('cuda'),
        'position_ids': position_ids.to('cuda'),
        'position_ids_second': position_ids_second.to('cuda')
    }


dataset = MyDataset(article_info, model)
dataloader = DataLoader(dataset, batch_size=196, collate_fn=collate_fn)

emb = []

model.eval()
for batch in tqdm(dataloader):
    with torch.no_grad():
        _, paper_embed_1 = model.bert.forward(
            **batch,
            output_all_encoded_layers=False,
            checkpoint_activations=False,
        )
    emb.append(paper_embed_1)
    del batch
    gc.collect()
    torch.cuda.empty_cache()

np.save('usr_data/oag_emb.npy', torch.concat(emb).cpu().numpy())
