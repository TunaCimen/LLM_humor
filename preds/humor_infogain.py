from openai import OpenAI
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics
from datasets import load_dataset

model = "gpt-4o-mini"
api_key = "apikey"
client = OpenAI(api_key=api_key)
def run_llm(prompt):
    response = client.responses.create(
        model = model,
        input = [
            {
                "role" : "user",
                "content" : [
                    {"type" : "input_text", "text" : prompt}
                ]
            },
        ],
    )
    return response.output[0].content[0].text



def get_json_from_generated_text(text):
    try:
        # extract substring between the first '{' and the last '}'
        start = text.find("{")
        end = text.rfind("}")
        
        if start == -1 or end == -1 or start >= end:
            print(f"Warning: No valid JSON brackets found in text")
            return {"Steps": []}
        
        json_str = text[start:end+1]
        json_obj = json.loads(json_str)
        
        return json_obj
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parsing failed - {e}")
        print(f"Problematic JSON text: {json_str[:200]}...")
        return {"Steps": []}
    except Exception as e:
        print(f"Warning: Unexpected error in JSON extraction - {e}")
        return {"Steps": []}


def llm_decompose_reasoning(llm_output, engine='gpt-4o'):
    try:
        prompt = f"""You are a pattern-following humor analyst for the New Yorker Caption Contest (NYCC).
Given a paragraph of reasoning that justifies why one caption is funnier than another, your task is to decompose the reasoning into individual steps.
Each step must include four components:
1. "id" : a unique number for the step
2. "planning" : a brief description of the main idea or reasoning move in this step. These should form a logically coherent sequence.
3. "knowledge" : the humor-related insight applied in this step (e.g., incongruity, irony, topicality, reference, stereotype, character behavior).
4. "step_text" : the sentence(s) from the input reasoning paragraph that support this step.

### Output Format:
Strictly follow the JSON structure below:

```json
{{
"Steps": [
    {{"id": 1, "planning": "Planning for step 1", "knowledge": "Humor concept or mechanism used in step 1", "step_text": "Sentence(s) from the input reasoning paragraph"}},
    {{"id": 2, "planning": "Planning for step 2", "knowledge": "Humor concept or mechanism used in step 2", "step_text": "Sentence(s) from the input reasoning paragraph"}},
    ...
]
}}
{llm_output}
"""
        res_text = run_llm(prompt)
        return get_json_from_generated_text(res_text)
    except Exception as e:
        print(f"Warning: LLM decomposition failed - {e}")
        return {"Steps": []}


def read_file(file_path):
    """
    Read and return the contents of the file at file_path.
    Returns None if the file cannot be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: file '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
    return None




file_name = "preds/ranking_qwen2_5_32b_thinking.json"


def decompose(file_name):
    save_file_name = file_name.split(".")[0] + "_decomposed.json"
    file = read_file(file_name)
    # Since this is already a proper JSON array, just parse it directly
    data = json.loads(file)
    predictions = list(map(lambda x: x["prediction"], data))
    predictions = list(filter(lambda x: x.get("ground_truth") == x.get("answer"), data))

    # Check how many entries are already processed
    try:
        existing_data = json.loads(read_file(save_file_name))
        n_already_processed = len(existing_data)
        print(f"Found {n_already_processed} already processed entries, skipping them...")
    except:
        existing_data = []
        n_already_processed = 0
        print("No existing output file found, starting from the beginning...")

    # Skip the already processed predictions
    remaining_predictions = predictions[n_already_processed:]
    remaining_data = data[n_already_processed:]

    print(f"Processing {len(remaining_predictions)} remaining predictions...")

    responses = []
    failed_count = 0
    total_processed = 0

    for i, pred in enumerate(tqdm(remaining_predictions, desc="Decomposing reasoning")):
        total_processed += 1
        try:
            decomposed = llm_decompose_reasoning(pred)
            
            # Check if decomposition was successful (has valid steps)
            if not decomposed.get("Steps") or len(decomposed["Steps"]) == 0:
                print(f"Warning: Empty or invalid decomposition for item {n_already_processed + i + 1}")
                failed_count += 1
                # Create a minimal entry to maintain indexing
                decomposed = {"Steps": [{"id": 1, "planning": "Failed to decompose", "knowledge": "Error", "step_text": "Decomposition failed"}]}
            
            responses.append(decomposed)
            
            # Create output entry for this new response
            output_entry = {
                "contest_number": remaining_data[i].get("contest_number",""),
                "answer": remaining_data[i].get("answer", ""),
                "ground_truth": remaining_data[i].get("ground_truth", ""),
                "steps": decomposed.get("Steps", [])
            }
            
            # Add to existing data
            existing_data.append(output_entry)
            
            # Write to JSON file after each processing (incremental save)
            with open(save_file_name, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write to disk immediately
                
            # Print progress every 10 items
            if total_processed % 10 == 0:
                print(f"Progress: {total_processed}/{len(remaining_predictions)} processed, {failed_count} failed")
                
        except Exception as e:
            print(f"Critical error processing item {n_already_processed + i + 1}: {e}")
            failed_count += 1
            
            # Create a minimal error entry to maintain indexing
            error_entry = {
                "contest_number": remaining_data[i].get("contest_number",""),
                "answer": remaining_data[i].get("answer", ""),
                "ground_truth": remaining_data[i].get("ground_truth", ""),
                "steps": [{"id": 1, "planning": "Critical error", "knowledge": "Error", "step_text": str(e)}]
            }
            
            existing_data.append(error_entry)
            
            # Still save the file to maintain progress
            try:
                with open(save_file_name, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()  # Force write to disk immediately
            except Exception as save_error:
                print(f"Failed to save after error: {save_error}")

    print(f"Completed! Total {len(existing_data)} entries saved to {save_file_name}.")
    print(f"Success rate: {((total_processed - failed_count) / total_processed * 100):.1f}% ({total_processed - failed_count}/{total_processed})")
    if failed_count > 0:
        print(f"Failed decompositions: {failed_count} (entries marked with error steps)")


#Calculate InfoGain
def calculate_info_gain(file_name):
    ds = load_dataset("jmhessel/newyorker_caption_contest", "ranking")
    info_gain_file = file_name.replace('.json', '_infoGain.json')
    test_data = ds["test"]
    file_name_ref = "preds/ranking_gemma3_thinking.json"
    with open(file_name_ref, 'r') as file:
        capt_ref = json.load(file)

    j = json.loads(read_file(file_name))
    all_scores = []
    
    for i, entry in enumerate(j):
        try:
            truth = entry["ground_truth"]
            answer = entry["answer"]
            contest_no = entry["contest_number"]
            
            # Skip if ground_truth doesn't match answer
            if truth != answer:
                print(f"Entry {i}: Skipping - ground_truth ({truth}) != answer ({answer})")
                all_scores.append(None)
                continue
            
            # Find all entries with this contest number
            contest_choices = []
            for test_entry in test_data:
                if test_entry["contest_number"] == contest_no:
                    contest_choices.append(test_entry)
            
            if not contest_choices:
                print(f"Entry {i}: No contest data found for contest {contest_no}")
                all_scores.append(None)
                continue
            
            # First appearance in file corresponds to last choice in dataset
            # So we need to reverse the index
            total_entries = len(contest_choices)
            # Count how many times this contest_no appears before current position
            appearances_before = sum(1 for prev_entry in j[:i] if prev_entry["contest_number"] == contest_no)
            # Reverse index: first appearance (0) -> last choice (total_entries-1)
            choice_index = total_entries - 1 - appearances_before
            
            if choice_index < 0 or choice_index >= total_entries:
                print(f"Entry {i}: Invalid choice index {choice_index} for contest {contest_no}")
                all_scores.append(None)
                continue
            
            target_entry = contest_choices[choice_index]
            caption_a_text = capt_ref[i]["caption_a"]
            caption_b_text = capt_ref[i]["caption_b"]
            
            print(f"Entry {i}: Contest {contest_no}, Choice {choice_index}")
            print(f"Caption A: {caption_a_text}")
            print(f"Caption B: {caption_b_text}")
            
            information_gain_scorer = metrics.InformationGainScorer(model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
            prompt_lines = [
                f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n",
                f"Contest number: {contest_no}.\n",
                f"I will provide you with two captions; one of them is deemed funnier by people.",
                f"Which of the following captions is funnier: A: {caption_a_text} B: {caption_b_text}?\n",
                f"A) {caption_a_text}\n",
                f"B) {caption_b_text}\n\n",
                f"Think step by step.",
                f"Please write your reasoning between <think> </think> tags,",
                f"and your final answer between <answer> </answer> tags.\n",
                f"Your answer should look like: <think> your reasoning here </think> <answer> your answer here </answer>"
            ]
            
            # Convert prompt_lines list to a single string
            question_text = "".join(prompt_lines)
            information_gain_scores = information_gain_scorer.forward(question_text, entry, truth)
            all_scores.append(information_gain_scores)
            print(f"Entry {i}: Information gain scores:", information_gain_scores)
            
            # Save to info gain file
            
            try:
                # Try to load existing data
                try:
                    with open(info_gain_file, 'r', encoding='utf-8') as f:
                        existing_info_gain = json.load(f)
                except FileNotFoundError:
                    existing_info_gain = []
                
                # Add current entry with info gain scores
                info_gain_entry = {
                    "contest_number": contest_no,
                    "answer": answer,
                    "ground_truth": truth,
                    "information_gain_scores": information_gain_scores
                }
                existing_info_gain.append(info_gain_entry)
                
                # Save updated data
                with open(info_gain_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_info_gain, f, indent=2, ensure_ascii=False)
                    
            except Exception as save_error:
                print(f"Error saving to info gain file: {save_error}")
            
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            all_scores.append(None)
            continue
    
    print(f"Processed {len(all_scores)} entries total")
    valid_scores = [score for score in all_scores if score is not None]
    print(f"Successfully processed {len(valid_scores)} entries")

#calculate_info_gain("preds/ranking_qwen2_5_32b_thinking_decomposed.json")
decompose("preds/ranking_gemma3_thinking.json")
