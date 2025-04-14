import json
import random
from datasets import load_dataset
from llms.utils import get_llm
from cotscope.segmentations.character import split_into_sentences
from study.test_segmenter import load_models

valid_classes = set([
    'back_tracking', 'verification', 'decomposition', 'backward_chaining',
    'error_recognition', 'reading_problem', 'conjecture', 'pausing', 'elimination',
    'abduction', 'induction', 'analogical', 'heuristic_recall', 'summarize', 'finalized',
    'followed_up', 'constraint_identification', 'elimination', 'formal_reasoning', 'deduction',
    'metacognition', 'divergent_thinking', 'thought_experiment'
])

available_categories = [
    # Error Handling
    '* back_tracking : belongs to error handling section, where the model realizes a path won\'t work and explicitly goes back to try a different approach. An example of backtracking is: "Let me try again" or "we need to try a different sequence". We want to mark instances where the chain-of-reasoning is abandoned and the model backtracks to a previous computation.',
    '* error_recognition : belongs to error handling section, but the step should be about recognizing an error was found and raised upon',
    
    # Problem Analysis
    "* verification : Asking itself whether this is correct. An example of an answer-verification step is: 'This sequence results in 1, which is not equal to 22' or 'Since 25 is not equal to 22'. We want to mark instances where the chain-of-reasoning explicitly checks the current result against the target number.",
    '* decomposition : where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I\'ll try to get close to {target//2}, then..."',
    '* backward_chaining : where the model starts from the target number and works backwards to the initial numbers. An example of backward-chaining when the target is 24 and the numbers are 12 and 2 is: "Let\'s work backwards from the target. 24/2 = 12. So, 12*2=24." and if the target is 22 and the numbers are 25 and 3 is: "Since the target is 22, and 22 + 3 = 25, ...".',
    '* reading_problem : where the step simply restates or paraphrases the problem without adding analysis. Example: "The problem asks us to find the sum of all even numbers in the sequence."',
    '* constraint_identification : actively analyzing the limitations or rules that define the solution space. Example: "This means we can only use each number once, and the operations must result in integers, which constrains our approach."',
    '* elimination : ruling out potential solutions or approaches. Example: "We can\'t use division here because it would result in a fraction, which isn\'t allowed."',
    
    # Reasoning Types
    '* abduction : finding most likely cause from the clues, often the most plausible conclusion given the incomplete information',
    '* induction : reasoning from a specific observation to general rules, its about drawing general patterns from specific instances',
    '* deduction : deriving specific conclusions from general principles or rules. Example: "Since all squares have equal sides and this shape has equal sides, it must be a square."',
    '* formal_reasoning : using symbolic logic, mathematical notation, or formal systems to solve a problem. Example: "Let x represent the unknown value. Then 2x + 5 = 15, which means x = 5."',
    
    # Knowledge Application
    '* heuristic_recall : recall using simple efficient rule, if a rule that work well in the past it will use it. Example: Let me recall the Angle Bisector Theorem. It states that the angle bisector divides the opposite side into segments proportional to the adjacent sides. So in triangle ABC, BD bisects angle ...',
    
    # Thought Process
    '* pausing : any mention of wait, hmm or filler words. Steps that do not involve any specific reasoning or classification action. Example: "Okay, let\'s see." But if this example contains context of other categories, then it should not be considered as pausing.',
    '* metacognition : when the model explicitly evaluates its own reasoning approach or confidence. Example: "I think I\'ve been approaching this problem incorrectly" or "I need to be more systematic here."',
    '* divergent_thinking : generating multiple possible approaches or solutions. Example: "There are several ways we could solve this: 1) Using the quadratic formula, 2) Completing the square, or 3) Factoring."',
    '* convergent_thinking : narrowing down multiple possibilities to find the best solution. Example: "Among these approaches, factoring would be most efficient because the coefficients suggest easy factors."',
    '* thought_experiment : using hypothetical scenarios to explore consequences. Example: "What if we tried adding these values instead of multiplying them? Let\'s see what happens..."',
    
    # Summary and Continuation
    '* summarize : summarize/reiterate the current progress made so far, it might just be a summary of what we have done so far, but have not concluded to the final answer',
    '* finalized : finalized the entire reasoning and usually at the end of the model. Usually it mentions words like "The final answer is..."',
    '* followed_up : the current step is only a followed up step to the existing step, previous step : Next, 144. Again, divide by 2 first:\n; Current step : 144 ÷ 2 = 72\n 72 ÷ 2 = 36. The arithmetic step is just a followup step of the previous step meaning the current step is the same purpose as previous one',

    # mathematics calculation
    '* mathematics_operation: the current steps does mathematics calculation. This is an additional steps you can add to the CLASS, meaning it should be added as well on top of the best matched class if it includes mathematics calculation.',

]


prompt = """You are tasked to label each step segment enclosed by <step_X> reasoning here </step_X> and determine what kind of reasoning chain does this particular steps belongs to.

Here's a list of classifcation steps which we want to predict:

%s

[PROBLEM]

%s

[REASONING]

%s


[OUTPUT FORMAT]

Make sure your outputs always in the following format, DO NOT USE MARKDOWN:

REASONING: <think step by step here>
OUTPUT: 
<step_1>...</step_1>
<step_2>...</step_2>
<step_3>...</step_3>
<step_4>...</step_4>
...



"""

# classifier, sentence_model, device = load_models('best_f1_sentence_pair_classifier.pt', 'intfloat/multilingual-e5-base')

def generate_from_existing_reasoning():
    subset = "deepseek-v3-r1-zero"
    dataset_name = "syntaxsynth/reasoning-conversations"
    model_name = "deepseek-ai/DeepSeek-V3"

    dataset_name = "cognitivecomputations/dolphin-r1"
    subset = "reasoning-flash"

    llm = get_llm(model_name, series="together")
    dataset = load_dataset(dataset_name, subset, split="train")
    added = set()
    with open('reasoning_labeling.jsonl', 'r') as f:
        for line in f:
            added.add(json.loads(line)['hash_id'])

    for idx, row in enumerate(dataset):
        if 'hash_id' in row:
            uid = "{}-{}".format(row['src'], row['hash_id'])
        else:
            uid = "{}-{}".format(subset, idx)

        if uid in added:
            continue

        prev_seg = None
        if 'instruction' in row:
            problem_str = row['instruction']
        else:
            problem_str = row['messages'][0]['content']

        reasonings = []
        for seg in split_into_sentences(row['reasoning']):
            print('[Current STEP]', seg)
            answers = []
            for _ in range(5):
                random.shuffle(available_categories)
                categories_str = '\n\n'.join(available_categories)
                input = prompt % (categories_str, problem_str, str(prev_seg), seg)
                res, res_info = llm(input, temperature=0.6)
                answers.append(res_info)
                print(res)
            print('---------')
            prev_seg = seg
            reasonings.append({
                'prev_seg': prev_seg,
                'seg': seg,
                'answers': answers
            })
        output = {
            'src': f'{dataset_name}@{subset}',
            'model': model_name,
            'problem': problem_str,
            'hash_id': uid,
            'reasonings': reasonings,
        }
        with open('reasoning_labeling.jsonl', 'a') as fout:
            fout.write(json.dumps(output)+'\n')
        print('=================')
        if idx > 10:
            break

def convert_sep_to_steps(text):
    """
    Converts text with <sep> separators to numbered steps with <step_n> tags.
    
    Args:
        text (str): The input text with <sep> separators
        
    Returns:
        str: The converted text with <step_n> tags
    """
    # Split the text by <sep> tags
    parts = text.split("<sep>")
    
    # Create the output with numbered steps
    result = []
    total_steps = 0
    for i, part in enumerate(parts, 1):
        # Remove leading/trailing whitespace but preserve internal formatting
        step_content = part.strip()
        if step_content:  # Only add non-empty steps
            result.append(f"<step_{i}>\n{step_content}\n</step_{i}>")
            total_steps = i
    
    # Join all steps with a newline between them
    return "\n".join(result), total_steps

def load_math_500():
    problems = []
    with open('../agent_scaling_study/logs/MATH-500/cot/Qwen2.5-Coder-32B-Instruct_v0.jsonl', 'r') as f:
        for line in f:
            problems.append(json.loads(line)['input'])
    data = []
    with open('metamath-qwen_gpt-4o-2024-11-20.jsonl', 'r') as f:
        for idx, line in enumerate(f):
            payload = json.loads(line)
            reasoning = payload['output'].strip()
            data.append({
                'reasoning': reasoning,
                'problem': problems[idx]
            })
    return data


def load_reasoning():
    dataset = load_dataset("syntaxsynth/reasoning-conversations", "deepseek-v3-r1", split="train")

    hash_id2data = {}
    with open('deepseek-r1_o3_mini_result.jsonl', 'r') as fin:
        for line in fin:
            payload = json.loads(line)
            hash_id2data[payload['hash_id']] = payload
    with open('deepseek-r1_gpt-4o-2024-11-20.jsonl', 'r') as fin:
        for line in fin:
            payload = json.loads(line)
            hash_id2data[payload['hash_id']] = payload

    data = []
    for row in dataset:
        if row['hash_id'] not in hash_id2data:
            break
        instruction = row['instruction']
        data.append({
            'problem': instruction,
            'reasoning': hash_id2data[row['hash_id']]['output']
        })
    return data

def load_reasoning_zero():
    dataset = load_dataset("syntaxsynth/reasoning-conversations", "deepseek-v3-r1-zero", split="train")

    hash_id2data = {}
    with open('deepseek-r1-zero_o3_mini_result.jsonl', 'r') as fin:
        for line in fin:
            payload = json.loads(line)
            hash_id2data[payload['hash_id']] = payload
    with open('deepseek-r1-zero_gpt-4o-2024-11-20.jsonl', 'r') as fin:
        for line in fin:
            payload = json.loads(line)
            hash_id2data[payload['hash_id']] = payload

    data = []
    for row in dataset:
        if row['hash_id'] not in hash_id2data:
            break
        instruction = row['instruction']
        data.append({
            'problem': instruction,
            'reasoning': hash_id2data[row['hash_id']]['output']
        })
    return data

def load_gemini_flash():
    dataset = load_dataset("cognitivecomputations/dolphin-r1", "reasoning-flash", split="train")

    hash_id2data = {}
    with open('gemini-flash-thinking_o3_mini_result.jsonl', 'r') as fin:
        for line in fin:
            payload = json.loads(line)
            hash_id2data[payload['hash_id']] = payload
    with open('gemini-flash-thinking_gpt-4o-2024-11-20.jsonl', 'r') as fin:
        for line in fin:
            payload = json.loads(line)
            hash_id2data[payload['hash_id']] = payload

    data = []
    for idx, row in enumerate(dataset):
        row['hash_id'] = idx
        if row['hash_id'] not in hash_id2data:
            break
        instruction = row['messages'][0]['content']
        data.append({
            'problem': instruction,
            'reasoning': hash_id2data[row['hash_id']]['output']
        })
    return data


def test_one_step():
    dataset = {
        'math_500': load_math_500(),
        'r1_reasoning': load_reasoning(),
        'r0_reasoning': load_reasoning_zero(),
        'gemini_flash': load_gemini_flash(),
    }
    from tqdm import tqdm
    dataset_name = 'r0_reasoning'
    target_data = dataset[dataset_name]

    # llm = get_llm("gpt-4o-2024-11-20", "openai")
    # llm = get_llm("gemini-2.5-pro-preview-03-25", "gemini")
    model_name = "deepseek-ai/DeepSeek-V3"
    llm = get_llm(model_name, series="together")
    added = set()
    # with open(dataset_name+'_deepseek_v3_labeling.jsonl', 'r') as f:
    #     for line in f:
    #         added.add(json.loads(line)['problem'])
    print(dataset_name)
    for row in tqdm(target_data):
        reasoning = row['reasoning']
        problem_str = row['problem']
        if problem_str in added:
            continue

        if reasoning.startswith('```'):
            reasoning = reasoning[4:]
        if reasoning.endswith('```'):
            reasoning = reasoning[:-4]
        # print()
        reasoning_with_step = convert_sep_to_steps(reasoning)[0]

        answers = []
        for _ in range(5):
            random.shuffle(available_categories)
            categories_str = '\n\n'.join(available_categories)
            input = prompt % (categories_str, problem_str, reasoning_with_step)
            res, res_info = llm(input, temperature=0.6, max_tokens=16384)
            answers.append(res_info)
            # print('--------------')
            # print(res)
        output = {
            'reasoning': reasoning,
            'problem': problem_str,
            'reasoning_with_step': reasoning_with_step,
            'answers': answers
        }
        with open(dataset_name+'_deepseek_v3_labeling.jsonl', 'a') as fout:
            fout.write(json.dumps(output)+'\n')

if __name__ == "__main__":
    # generate_from_existing_reasoning()
    test_one_step()