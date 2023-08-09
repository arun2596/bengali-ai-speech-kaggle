import os
import pandas as pd 
from data_loader import DatasetRetriever, DataCollatorCTCWithPadding

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC,  EarlyStoppingCallback
import configs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "arijitx/wav2vec2-xls-r-300m-bengali",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    bos_token="<s>",
    eos_token="</s>",
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=configs.SR,
    padding_value=0.0,
    padding_side="right",
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)




train_annotations_path  = os.path.join(configs.ANNOTATION_FOLDER, 'train.csv')
df = pd.read_csv(train_annotations_path)

#getting train and valid dfs
train_df = df[df['split']=='train'].reset_index(drop=True)
valid_df = df[df['split']=='valid'].reset_index(drop=True)


train_df = train_df[train_df['id'].apply(lambda x: exist_dict[x])]
valid_df = valid_df[valid_df['id'].apply(lambda x: exist_dict[x])]
print(train_df.shape, valid_df.shape)

#creating train and validation dataset
train_dataset = DatasetRetriever(train_df, processor)
valid_dataset = DatasetRetriever(valid_df, processor)
# print(valid_df.head())

# Loading model.
model = Wav2Vec2ForCTC.from_pretrained(
    "arijitx/wav2vec2-xls-r-300m-bengali", 
    ignore_mismatched_sizes=False,
    attention_dropout=configs.w2v_config.attention_dropout,
    hidden_dropout=configs.w2v_config.hidden_dropout,
    feat_proj_dropout=configs.w2v_config.feat_proj_dropout,
    mask_time_prob=configs.w2v_config.mask_time_prob,
    layerdrop=configs.w2v_config.layerdrop,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# Freezing encoder layers.
model.freeze_feature_encoder()

# Printing stats.
total_param = sum(p.numel() for p in model.parameters())
trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total_param = {total_param}")
print(f"trainable = {trainable_param}")

data_collator = DataCollatorCTCWithPadding(processor, padding=True)

from utils import compute_metrics_wrapper
from transformers import Trainer

compute_metrics = compute_metrics_wrapper(processor)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=configs.w2v_config.trainer,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,   
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=configs.w2v_config.early_stopping_patience)],
)

trainer.train()