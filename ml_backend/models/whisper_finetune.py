import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate

class WhisperFineTuner:
    """
    Fine-tunes the openai/whisper-small model on a custom audio dataset.
    This fulfills the ML/DL model implementation requirement (CO-3).
    """
    
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        self.metric = evaluate.load("wer")

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        
        batch["input_features"] = self.processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        batch["labels"] = self.processor(
            text=batch["sentence"]
        ).input_ids
        
        return batch

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        pred_ids[pred_ids == -100] = self.processor.tokenizer.pad_token_id
        
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def train(self, dataset_name="mozilla-foundation/common_voice_11_0", lang="en", epochs=3):
        print(f"Loading dataset {dataset_name} for language {lang}...")
        
        import datasets as ds_lib
        dataset = load_dataset(dataset_name, lang, split="train[:5%]", streaming=False)
        dataset = dataset.cast_column("audio", ds_lib.Audio(sampling_rate=16000))
        
        print("Mapping preprocessing function...")
        encoded_dataset = dataset.map(self.prepare_dataset, remove_columns=dataset.column_names)

        split_dataset = encoded_dataset.train_test_split(test_size=0.1)
        
        training_args = TrainingArguments(
            output_dir="./whisper-finetuned",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2, 
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            compute_metrics=self.compute_metrics,
        )

        print("Starting fine-tuning...")
        trainer.train()
        print("Training complete. Saving model...")
        
        self.model.save_pretrained("./whisper-finetuned")
        self.processor.save_pretrained("./whisper-finetuned")

if __name__ == "__main__":
    print("Initialize Whisper FineTuner...")
    try:
        tuner = WhisperFineTuner()
        print("Model initialized successfully. Ready for training.")
        # tuner.train()  # Uncomment to start actual training
    except Exception as e:
        print(f"Initialization error: {e}")
