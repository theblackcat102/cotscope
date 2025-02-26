from datasets import load_dataset
from sentence_transformers import CrossEncoder
from cotscope.segmentations.character import split_into_sentences
from openai import OpenAI

def model_base_nli():
    model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    dataset = load_dataset("cognitivecomputations/dolphin-r1", "reasoning-deepseek", split="train")
    for idx, row in enumerate(dataset):
        prev_sent = None
        for sentence in split_into_sentences(row['reasoning']):
            if prev_sent is None:
                prev_sent = sentence
                continue
            print(sentence)
            scores = model.predict([(prev_sent, sentence)])
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [(label_mapping[score_max], scores[0, score_max]) for score_max in scores.argmax(axis=1)]
            print(labels)
            prev_sent = sentence
        print('---------Next sample---------')

def llm_base_nli():

    system_prompt = """\
You are an AI assistant whose task is to classify **how each new sentence in a text relates to the previous sentence**. For each pair of consecutive sentences, you will do the following:

1. **Choose exactly one** relationship label from the list below.  
2. Provide a **brief explanation** (1–2 lines) of why you chose that label.

Here are the **relationship labels** and their definitions:

1. **Elaboration**  
   - The new sentence adds more detail or expands on the previous sentence.  

2. **Explanation**  
   - The new sentence clarifies or interprets the idea in the previous sentence.  

3. **Evidence**  
   - The new sentence offers data, facts, or proof supporting the previous sentence.  

4. **Example**  
   - The new sentence illustrates the previous sentence with a specific case or instance.  

5. **Summary**  
   - The new sentence restates the previous sentence in a more condensed form.  

6. **Paraphrase**  
   - The new sentence repeats the same idea as the previous sentence but in different words.  

7. **Contradiction**  
   - The new sentence explicitly opposes or conflicts with the previous sentence.  

8. **Conclusion**  
   - The new sentence draws a conclusion or inference from the previous sentence(s).  

9. **Transition**  
   - The new sentence signals a topic shift or indicates a next step in the reasoning.  

10. **Self-Reflection**  
   - The new sentence is the speaker “thinking aloud,” planning, or evaluating actions (meta-cognition).  

11. **Instruction**  
   - The new sentence issues a directive or command (e.g., “Make sure to…,” “Check if…”).  

12. **Other**  
   - For any relationship that doesn’t fit the above categories.

You will produce your output **pair by pair** in the following format:

```
Sentence #i → Sentence #(i+1):
Explanation: <BRIEF REASONING>
Relationship: <CHOSEN LABEL>
"""

    samples = """\
Let me start by identifying the main points. → First, Maria is informing Pierre about her invitation to lecture at a conference on Baroque Art in Paris.:
Relationship: Transition
Explanation: The speaker moves from stating the intention to identify main points to actually mentioning Maria’s invitation.

First, Maria is informing Pierre about her invitation to lecture at a conference on Baroque Art in Paris. → She expresses excitement about visiting Paris and reconnecting with him.:
Relationship: Elaboration
Explanation: Sentence 3 builds on the details from the invitation by expressing excitement about visiting Paris.
"""

    import json
    client = OpenAI()
    model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    dataset = load_dataset("cognitivecomputations/dolphin-r1", "reasoning-deepseek", split="train")

    with open('ds-r1-math-train.jsonl', 'r') as f:
        # for line in f:
        for idx, row in enumerate(dataset):
            # row = json.loads(line)
            prev_sent = None
            for sentence in split_into_sentences(row['reasoning']):
                if prev_sent is None:
                    prev_sent = sentence
                    continue
                print(sentence)
                response = client.chat.completions.create(
                    model="gpt-4o",  # or another supported model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Let me start by identifying the main points. → First, Maria is informing Pierre about her invitation to lecture at a conference on Baroque Art in Paris.:"},
                        {"role": "assistant", "content": "Explanation: The speaker moves from stating the intention to identify main points to actually mentioning Maria’s invitation.\nRelationship: Transition"},
                        {"role": "user", "content": "First, Maria is informing Pierre about her invitation to lecture at a conference on Baroque Art in Paris. → She expresses excitement about visiting Paris and reconnecting with him.:"},
                        {"role": "assistant", "content": "Explanation: New sentence builds on the details from the invitation by expressing excitement about visiting Paris.\nRelationship: Elaboration"},
                        {"role": "user", "content": "{} → {}:".format(prev_sent, sentence)}
                    ],
                    temperature=0.7,
                    max_completion_tokens=5120
                )

                # Extract and print the assistant's reply
                assistant_reply = response.choices[0].message.content
                print(assistant_reply)
                print('-----')
                prev_sent = sentence
            print('---------Next sample---------')



if __name__ == "__main__":
    model_base_nli()