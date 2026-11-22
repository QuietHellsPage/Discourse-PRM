"""
Main training script that contains all the logic
"""

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from config.constants import DATASET_NAME, MAX_LENGTH, MODEL_NAME, OUTPUT_PATH, SEED
from discourse_prm.data.loader import obtain_dataset
from discourse_prm.data.prepare import prepare_dataset
from discourse_prm.metrics import compute_metrics
from discourse_prm.models.loader import obtain_model, obtain_tokenizer


def train(resume_from_checkpoint: bool = False) -> None:
    """
    Main training method

    Args:
      resume_from_checkpoint (bool): Whether should start from the existing checkpoint.
    """
    set_seed(SEED)
    tokenizer = obtain_tokenizer(MODEL_NAME)
    model = obtain_model(MODEL_NAME)
    dataset = prepare_dataset(obtain_dataset(DATASET_NAME), tokenizer, MAX_LENGTH)
    validation = dataset["validation"].shuffle(seed=SEED)
    validation = validation.select(range(min(1000, len(validation))))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_PATH),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            max_steps=10000,
            logging_strategy="steps",
            logging_steps=25,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            dataloader_pin_memory=False,
            report_to="tensorboard",
            seed=SEED,
            warmup_steps=300,
        ),
        train_dataset=dataset["train"],
        eval_dataset=validation,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(str(OUTPUT_PATH / "best"))
    trainer.evaluate(dataset["test"], metric_key_prefix="test")
