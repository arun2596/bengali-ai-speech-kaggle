import torch
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
import librosa
import torchaudio
import torchaudio.transforms as tat

import pandas as pd
import configs
import os


from typing import Dict, List, Tuple, Any, Union, Optional

class DatasetRetriever(Dataset):
    def __init__(self, data, processor, data_folder = configs.TRAIN_DATA_FOLDER, is_train=True):
        
        #data from annotations
        self.data = data
        self.data_folder = data_folder
        self.audio_ids = self.data['id']
        self.sentences = self.data['sentence']

        #loaders and transforms
        self.processor = processor

        # self.resampler = tat.Resample(32000, configs.SR)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        sentence = self.sentences[ind]
        audio_id = self.audio_ids[ind]

        audio_path = os.path.join(self.data_folder, str(audio_id) + '.mp3')
        
        #load the audio here - maybe load the full audio for augmentation and then downsample
        wave = librosa.load(audio_path, sr=configs.SR)[0]
        input_values = self.processor(wave, sampling_rate=configs.SR).input_values[0]

        input_length = len(input_values)
        with self.processor.as_target_processor():
            labels = self.processor(sentence).input_ids

        return {
            'input_values':input_values,
            'input_length':input_length,
            'labels':labels
        }



    def make_loader(data, batch_size):
        pass


class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length

#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     ## get sequence lengths
#     lengths = torch.tensor([ t.shape[0] for t in batch ])
#     ## padd
#     batch = [ torch.Tensor(t) for t in batch ]
#     batch = torch.nn.utils.rnn.pad_sequence(batch)
#     ## compute mask
#     mask = (batch != 0)
#     return batch, lengths, mask

