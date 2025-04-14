import torch
import argparse
import logging
from transformers import AutoTokenizer, ModernBertForTokenClassification
from typing import List, Optional, Union

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def predict_step_splits(
        text: str, 
        model: ModernBertForTokenClassification, 
        tokenizer: AutoTokenizer, 
        sep_token: str = "[SEP]", 
        device: Optional[torch.device] = None
    ) -> List[int]:
    """
    Predict whether each [SEP] token should split into a new step.
    
    Args:
        text: Input text with [SEP] tokens
        model: Trained ModernBertForTokenClassification model
        tokenizer: Tokenizer for the model
        sep_token: Token used to separate text segments
        device: Device to run inference on (defaults to CUDA if available)
        
    Returns:
        List of binary predictions (0 or 1) for each [SEP] token
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
    
    # Find SEP tokens
    input_ids = inputs["input_ids"][0]
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    
    # Get prediction for each SEP token
    sep_predictions = [predictions[0, pos].item() for pos in sep_positions]
    return sep_predictions

def format_steps(text: str, split_predictions: List[int], sep_token: str = "[SEP]") -> List[str]:
    """
    Format text into steps based on the split predictions.
    
    Args:
        text: Input text with [SEP] tokens
        split_predictions: Binary predictions for each [SEP] token
        sep_token: Token used to separate text segments
        
    Returns:
        List of formatted steps
    """
    text_parts = text.split(sep_token)
    formatted_steps = []
    current_step = text_parts[0]
    
    for i, pred in enumerate(split_predictions):
        if i + 1 < len(text_parts):
            if pred == 1:  # This SEP should be a step break
                formatted_steps.append(current_step.strip())
                current_step = text_parts[i + 1]
            else:  # This SEP should not be a step break
                current_step += " " + text_parts[i + 1]+'\n'
    
    # Add the last step
    if current_step.strip():
        formatted_steps.append(current_step.strip())
    
    return formatted_steps

def main():
    parser = argparse.ArgumentParser(description="Use a trained model to predict step splits")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to a text file containing input to process",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the output steps",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Direct text input to process (alternative to input_file)",
    )
    parser.add_argument(
        "--sep_token",
        type=str,
        default="[SEP]",
        help="Token used to separate text segments",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_file and not args.text:
        raise ValueError("Either --input_file or --text must be provided")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = ModernBertForTokenClassification.from_pretrained(args.model_path)
    
    # Get input text
    if args.input_file:
        logger.info(f"Reading input from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    
    # Get predictions
    logger.info("Running inference...")
    text = text.replace('\n\n', args.sep_token).replace('\n', args.sep_token)
    split_predictions = predict_step_splits(text, model, tokenizer, args.sep_token)
    
    # Format steps
    formatted_steps = format_steps(text, split_predictions, args.sep_token)
    
    # Print or save output
    if args.output_file:
        logger.info(f"Saving output to {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, step in enumerate(formatted_steps):
                f.write(f"Step {i+1}: {step}\n\n")
    else:
        print("\n===== Formatted Steps =====\n")
        for i, step in enumerate(formatted_steps):
            print(f"Step {i+1}: {step}\n")
    
    logger.info(f"Successfully processed text and identified {len(formatted_steps)} steps")

if __name__ == "__main__":
    main()