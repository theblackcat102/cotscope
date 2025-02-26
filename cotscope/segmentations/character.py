import re

def split_into_sentences(reasoning_text):
    """
    Splits a reasoning string into sentences based on punctuation boundaries.
    It looks for periods, exclamation points, or question marks followed by whitespace.
    """
    # The regex pattern (?<=[.!?])\s+ matches any whitespace following a sentence-ending punctuation.
    sentences = re.split(r'(?<=[.!?。！？])\s+', reasoning_text.strip())
    return sentences
