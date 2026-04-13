import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import evaluate
import numpy as np

class BartSummarizerFineTuner:
    """
    Fine-tunes the facebook/bart-large-cnn model for text summarization using LoRA.
    This fulfills the ML/DL model implementation requirement (CO-3).
    """
    
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()
        
        self.metric = evaluate.load("rouge")

    def preprocess_function(self, dataset_batch):
        inputs = dataset_batch["dialogue"]
        targets = dataset_batch["summary"]
        
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_preds):
        import nltk
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    def train(self, dataset_name="samsum", epochs=3):
        print(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name)
        
        print("Preprocessing dataset...")
        tokenized_datasets = dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=dataset["train"].column_names
        )

        training_args = TrainingArguments(
            output_dir="./bart-lora-summarizer",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
        )

        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        print("Starting LoRA fine-tuning for BART...")
        trainer.train()
        print("Fine-tuning complete. Saving adapter model...")
        
        trainer.model.save_pretrained("./bart-lora-summarizer")
        self.tokenizer.save_pretrained("./bart-lora-summarizer")


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    
    print("Initialize BART LoRA FineTuner...")
    try:
        tuner = BartSummarizerFineTuner()
        print("Model initialized successfully with LoRA. Ready for training.")
        # tuner.train(epochs=1)  # Uncomment to start actual training
    except Exception as e:
        print(f"Initialization error: {e}")
