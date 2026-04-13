import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import numpy as np

class BertNERFineTuner:
    """
    Fine-tunes dbmdz/bert-large-cased-finetuned-conll03-english for Named Entity Recognition.
    In the context of MeetingMind, this helps extract Action Items, Deadlines, and Assignees.
    Fulfills ML/DL Application (CO-3).
    """
    
    def __init__(self, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )
        self.metric = evaluate.load("seqeval")
        self.label_list = []

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, dataset_name="conll2003", epochs=3):
        print(f"Loading NER dataset {dataset_name}...")
        dataset = load_dataset(dataset_name)
        
        self.label_list = dataset["train"].features["ner_tags"].feature.names
        
        print("Tokenizing and aligning labels...")
        tokenized_datasets = dataset.map(
            self.tokenize_and_align_labels, 
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./bert-ner-meeting",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=3,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        print("Starting BERT NER fine-tuning...")
        trainer.train()
        print("Training complete. Saving NER model...")
        
        trainer.save_model("./bert-ner-meeting")
        self.tokenizer.save_pretrained("./bert-ner-meeting")

if __name__ == "__main__":
    print("Initialize BERT NER FineTuner...")
    try:
        tuner = BertNERFineTuner()
        print("Model initialized successfully. Ready for training.")
        # tuner.train(epochs=1)  # Uncomment to start actual training
    except Exception as e:
        print(f"Initialization error: {e}")
