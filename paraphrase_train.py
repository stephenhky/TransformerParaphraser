
import os
from argparse import ArgumentParser


from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments
)


def modelTrainer(text_path, epochs, output_dir, model='gpt2', tokenizer='gpt2', batch_size=8):
    model = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_path,
        block_size=256
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_gpu_train_batch_size=batch_size,
        warmup_steps=500,
        save_steps=2000,
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model()


