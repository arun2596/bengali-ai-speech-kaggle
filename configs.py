SR = 16000 #wav2vec2 requires samplerate 16000
ANNOTATION_FOLDER = '/home/users/ssriniv2/learn-comp/data'
TRAIN_DATA_FOLDER = '/home/users/ssriniv2/learn-comp/data/train_mp3s/'

import os
from transformers import TrainingArguments

class w2v_config:
   # Dropout configs for pretrained wav2vec2 model.
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.1
    mask_time_prob = 0.05
    layerdrop = 0.1
        
    # Early stopping.
    early_stopping_patience = 10

    # Trainer arugments.
    trainer = TrainingArguments(
      output_dir="home/users/ssriniv2/learn-comp/model-out/output/8sept-001-arjith-300-fulldata-cosine-lr3e-5-ep3",
      group_by_length=False,
      per_device_train_batch_size=12,
      per_device_eval_batch_size=12,
      ddp_find_unused_parameters = True,
      gradient_accumulation_steps=1,
      evaluation_strategy="steps",
      num_train_epochs=3,
      gradient_checkpointing=False,
      fp16=True,
      save_steps=5000,
      eval_steps=5000,
      # eval_accumulation_steps = 256,
      logging_steps=5000,
      learning_rate=3e-5,
      dataloader_num_workers=24,
      warmup_steps=1000,
      save_total_limit=10,
      push_to_hub=False,
      run_name="8sept-001-arjith-300-fulldata-cosine-lr3e-5-ep3",
      load_best_model_at_end=True,
      lr_scheduler_type="cosine",
      resume_from_checkpoint=False,
    )