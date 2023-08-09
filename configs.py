SR = 16000 #wav2vec2 requires samplerate 16000
ANNOTATION_FOLDER = 'data/raw/'
TRAIN_DATA_FOLDER = 'data/raw/train_mp3s/'

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
      output_dir="model/output/run-001-wav2vec2-fulldata-cosine-lr3e-5",
      group_by_length=False,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      gradient_accumulation_steps=1,
      evaluation_strategy="steps",
      num_train_epochs=10,
      gradient_checkpointing=True,
      fp16=True,
      save_steps=400,
      eval_steps=400,
      logging_steps=400,
      learning_rate=3e-5,
      dataloader_num_workers=os.cpu_count(),
      warmup_steps=300,
      save_total_limit=10,
      push_to_hub=False,
      run_name="run-001-wav2vec2-fulldata-cosine-lr3e-5",
      load_best_model_at_end=True,
      lr_scheduler_type="cosine",
      resume_from_checkpoint=True,
    )