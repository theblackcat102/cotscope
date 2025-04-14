import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    ModernBertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class StepSplitDataset(Dataset):
    """Dataset for step splitting token classification"""
    
    def __init__(self, sequences, labels, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        split_labels = self.labels[idx]
        
        # Tokenize the input
        inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the tokenizer
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        
        # Find positions of SEP tokens (either using 50282 as in example or from tokenizer)
        input_ids = inputs["input_ids"]
        sep_positions = (input_ids == self.sep_token_id).nonzero(as_tuple=True)[0]
        
        # Create labels array with the same shape as input_ids, filled with ignore index (-100)
        labels_ids = torch.full_like(input_ids, -100)
        
        # Assign labels to SEP token positions
        for i, pos in enumerate(sep_positions):
            if i < len(split_labels):
                labels_ids[pos] = split_labels[i]
            else:
                # Handle case where there are more SEP tokens than labels (e.g., due to truncation)
                break
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])),
            "labels": labels_ids
        }

def preprocess_data(jsonl_files, sep_token="[SEP]"):
    """
    Process JSONL files and extract sequences with labels for token classification.
    """
    data = []

    for jsonl_filename in tqdm(glob.glob(jsonl_files), desc="Processing JSONL files"):
        with open(jsonl_filename, 'r') as f:
            for line in f:
                try:
                    payload = json.loads(line)
                    
                    if '<sep>' not in payload['output'] or '\n' not in payload['output']:
                        continue
                    
                    labels = []
                    prev = None
                    for segment in payload['output'].split('<sep>'):
                        if len(segment.strip()) == 0:
                            continue
                            
                        if prev is not None:
                            curr = [s for s in segment.split('\n') if len(s)][0]
                            labels.append(([prev, curr], 1))  # 1 means split into new step
                            prev = None

                        for chunk in segment.split('\n'):
                            if len(chunk) == 0:
                                continue
                            if prev:
                                labels.append(([prev, chunk], 0))  # 0 means don't split
                            prev = chunk
                    
                    # Process labels to create sequence and split_labels
                    sequence = ''
                    split_labels = []
                    for (pair, label) in labels:
                        if len(sequence) == 0:
                            sequence = sep_token.join(pair)
                            split_labels.append(label)
                        else:
                            sequence += sep_token + pair[-1]
                            split_labels.append(label)
                    
                    if len(sequence) > 0 and len(split_labels) > 0:
                        data.append((sequence, split_labels))
                except Exception as e:
                    logger.warning(f"Error processing line: {e}")
                    continue
    
    return data

def compute_metrics(pred):
    """
    Compute metrics for token classification evaluation.
    """
    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids
    
    # Only consider positions where label is not -100 (ignore index)
    mask = labels != -100
    labels_filtered = labels[mask]
    predictions_filtered = predictions[mask]
    
    if len(labels_filtered) == 0:
        return {
            'accuracy': 0,
            'f1': 0,
            'precision': 0,
            'recall': 0
        }
    
    accuracy = accuracy_score(labels_filtered, predictions_filtered)
    f1 = f1_score(labels_filtered, predictions_filtered, average='weighted')
    precision = precision_score(labels_filtered, predictions_filtered, average='weighted')
    recall = recall_score(labels_filtered, predictions_filtered, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(args):
    """Main training function"""
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Process data
    logger.info(f"Processing data from {args.data_path}")
    raw_data = preprocess_data(args.data_path)
    
    if len(raw_data) == 0:
        logger.error("No valid data found. Check your data files and format.")
        return
    
    logger.info(f"Found {len(raw_data)} valid sequences for training.")
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        raw_data, 
        test_size=args.validation_split, 
        random_state=args.seed
    )
    
    logger.info(f"Training on {len(train_data)} sequences, validating on {len(val_data)} sequences")
    
    # Extract sequences and labels
    train_sequences, train_labels = zip(*train_data)
    val_sequences, val_labels = zip(*val_data)
    
    # Create datasets
    train_dataset = StepSplitDataset(
        train_sequences, train_labels, tokenizer, args.max_seq_length
    )
    val_dataset = StepSplitDataset(
        val_sequences, val_labels, tokenizer, args.max_seq_length
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model = ModernBertForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2  # Binary classification: 0 (no split) or 1 (split)
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=5,
        seed=args.seed,
        report_to="wandb",
        dataloader_num_workers=args.num_workers,
        bf16=args.bf16,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    model_save_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Final model saved to {model_save_path}")
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    eval_results = trainer.evaluate()
    
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)

    logger.info(f"Training completed successfully")
    return model, tokenizer, eval_results

def parse_args():
    parser = argparse.ArgumentParser(description="Train a token classification model for step splitting")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL files with training data (can include wildcards, e.g., 'data/math-normal*.jsonl')",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./step_classifier_output",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=5.0,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Linear warmup over warmup_steps",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay if we apply some",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Weight decay if we apply some",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Percentage of training data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision training",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)