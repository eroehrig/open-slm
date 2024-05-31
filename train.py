# poetry run python train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback
from datasets import load_dataset
import datetime
import os
from utils import reconstruct_codec, flat_codec

BASE_MODEL = "BEE-spoke-data/smol_llama-101M-GQA"
EXPERIMENT_NAME = "try_again_with_full_ds"
MAX_LEN = 1024



# load model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
tokenizer.padding_side = "left"
model.resize_token_embeddings(len(tokenizer))
SEP = tokenizer("[audio]")["input_ids"][1:]

# load data
dataset = load_dataset("blanchon/snac_llm_parler_tts")["train"] 
#dataset = load_dataset("blanchon/snac_llm_parler_tts", split='train[0:100]' ) # to only get a subset for testing
dataset = dataset.train_test_split(test_size=0.3, seed=42)

def prepare_sample(sample):

    input_ids = tokenizer(sample["text"], padding=False, truncation=True, max_length=512-len(SEP))["input_ids"]+SEP
    target_ids = [int(t) for t in sample["snac24khz"].split(" ")][:512]
    labels = [-100] * len(input_ids) + target_ids
    return {"input_ids": input_ids+target_ids, "labels": labels}

tokenized_train_dataset = dataset["train"].map(prepare_sample, batched=False)
tokenized_val_dataset = dataset["test"].map(prepare_sample, batched=False)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  #for dynamic padding



# train
experiment = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M-") +EXPERIMENT_NAME +"based_on_"+BASE_MODEL
folder = f"./results/{experiment}"
os.makedirs(folder, exist_ok=False)

# for logging 
import soundfile as sf
from snac import SNAC

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac = snac.cuda()

class SaveCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        input_ids = tokenizer(dataset["test"][0]["text"],  padding=False, truncation=True, max_length=512-len(SEP))["input_ids"]+SEP

        with torch.no_grad():
            outputs = model.generate(torch.tensor([input_ids]).to("cuda"), max_length=MAX_LEN, pad_token_id=tokenizer.eos_token_id, temperature=0)
            outputs = outputs[0][len(input_ids):]

        try:
            codes = reconstruct_codec(outputs)
            audio_hat = snac.decode(codes) # problem with the wrong indices this breaks CUDA
            sf.write(f"{folder}/step_{state.global_step}.wav", audio_hat.cpu().detach().numpy().squeeze(), 24000)
        except Exception as e:
            print(f"Failed to create audio from created snac tokens due to {e}")
            #print("input_ids:", input_ids)
            #print("outputs:", outputs)
            #print("codes:", codes)


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=folder,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=3e-5,
        per_device_train_batch_size=30,
        num_train_epochs=10,  
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir=folder,
        logging_steps=10,
        save_steps=5000,
        save_total_limit=20,
        fp16=True,  
        lr_scheduler_type="cosine",
        warmup_steps=500  # Number of steps to perform learning rate warm-up
    ),
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    callbacks=[SaveCallback]
)

trainer.train()
trainer.save_model(f"{folder}/trained_model")

# loss ap epoch 0.23 -> 6.06