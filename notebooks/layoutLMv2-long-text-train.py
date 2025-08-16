from ast import arg
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_metric, load_from_disk
import os
import numpy as np
import torch
import warnings
import argparse
from PIL import Image

warnings.filterwarnings('ignore')

# -- Fix for PyArrow deprecated PyExtensionType deserialization --
try:
    import pyarrow as pa
    pa.PyExtensionType.set_auto_load(True)
    print("✅ Enabled pyarrow.PyExtensionType auto_load")
except Exception as e:
    print("⚠️ Could not enable pyarrow auto_load:", e)

def preprocess_data_with_overflow(examples, processor, max_length=512, stride=50):
    """
    Preprocess data handling overflow tokens and proper label alignment
    """
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples['words']
    boxes = examples['bboxes']
    word_labels = examples['ner_tags']
    
    # Process with overflow
    encoded_inputs = processor(
        images, 
        words, 
        boxes=boxes, 
        word_labels=word_labels,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        stride=stride
    )
    
    # Get mappings
    offset_mapping = encoded_inputs.pop('offset_mapping')
    overflow_to_sample_mapping = encoded_inputs.pop('overflow_to_sample_mapping')
    
    # Handle labels for overflowing tokens
    if 'labels' in encoded_inputs:
        labels = encoded_inputs['labels']
        # The processor should handle label alignment automatically
        # but let's ensure consistency
        encoded_inputs['labels'] = labels
    
    # Add sample mapping for evaluation
    encoded_inputs['overflow_to_sample_mapping'] = overflow_to_sample_mapping
    
    return encoded_inputs

class CustomTrainer(Trainer):
    """
    Custom trainer to handle predictions from multiple chunks of the same document
    """
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction step to handle overflow mapping
        """
        # Remove overflow mapping from inputs before model forward
        overflow_mapping = inputs.pop('overflow_to_sample_mapping', None)
        
        # Call parent prediction step
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        
        # Add overflow mapping back for post-processing
        if overflow_mapping is not None:
            inputs['overflow_to_sample_mapping'] = overflow_mapping
            
        return loss, logits, labels

def compute_metrics_with_overflow(eval_pred, id2label, processor):
    """
    Compute metrics handling multiple chunks per document
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Group predictions by original sample
    sample_predictions = {}
    sample_labels = {}
    
    # This is a simplified version - you might need to implement
    # proper aggregation of overlapping predictions
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        sample_id = i  # You'd need actual sample mapping here
        
        if sample_id not in sample_predictions:
            sample_predictions[sample_id] = []
            sample_labels[sample_id] = []
            
        # Remove ignored index (special tokens)
        valid_predictions = [
            id2label[p] for (p, l) in zip(pred, label) if l != -100
        ]
        valid_labels = [
            id2label[l] for (p, l) in zip(pred, label) if l != -100
        ]
        
        sample_predictions[sample_id].extend(valid_predictions)
        sample_labels[sample_id].extend(valid_labels)
    
    # Flatten for seqeval
    all_predictions = list(sample_predictions.values())
    all_labels = list(sample_labels.values())
    
    metric = load_metric("seqeval")
    results = metric.compute(predictions=all_predictions, references=all_labels)
    
    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    
    return final_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=2)  # Reduced due to longer sequences
    parser.add_argument("--eval_batch_size", type=int, default=2)   # Reduced due to longer sequences
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=str, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=50)
    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_dir", type=str)

    args, _ = parser.parse_known_args()
    EPOCHS = args.epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VALID_BATCH_SIZE = args.eval_batch_size
    LEARNING_RATE = float(args.learning_rate)
    LR_SCHEDULER_TYPE = args.lr_scheduler_type
    WARMUP_RATIO = float(args.warmup_ratio)
    MAX_LENGTH = args.max_length
    STRIDE = args.stride

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    train_dataset = load_from_disk(f'{args.data_dir}train_split')
    valid_dataset = load_from_disk(f'{args.data_dir}eval_split')
    
    # Prepare model labels
    labels = train_dataset.features["labels"].feature.names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(
        'microsoft/layoutlmv2-base-uncased',
        num_labels=len(label2id)
    )
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')
    
    # Set id2label and label2id 
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Apply preprocessing to datasets
    # Note: You'll need to implement this based on your dataset structure
    # train_dataset = train_dataset.map(
    #     lambda examples: preprocess_data_with_overflow(
    #         examples, processor, MAX_LENGTH, STRIDE
    #     ),
    #     batched=True,
    #     remove_columns=train_dataset.column_names
    # )
    
    # valid_dataset = valid_dataset.map(
    #     lambda examples: preprocess_data_with_overflow(
    #         examples, processor, MAX_LENGTH, STRIDE
    #     ),
    #     batched=True,
    #     remove_columns=valid_dataset.column_names
    # )

    def compute_metrics(p):
        """Simplified compute_metrics - you may want to use compute_metrics_with_overflow"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric = load_metric("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results

    from transformers import TrainingArguments as HFTrainingArguments
    
    training_args = HFTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        metric_for_best_model="overall_f1",
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_strategy="epoch",
        gradient_accumulation_steps=2,  # Helps with smaller batch sizes
        dataloader_pin_memory=False,    # Can help with memory issues
    )

    # Use custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_result = trainer.evaluate(eval_dataset=valid_dataset)

    # writes eval result to file which can be accessed later in s3 output
    print(f"***** Eval results *****")
    for key, value in sorted(eval_result.items()):
        print(f"{key} = {value}\n")
    
    trainer.save_model(args.output_dir)