import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

def extract_reasoning_and_answer(text):
    reasoning_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match    = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer    = answer_match.group(1).strip() if answer_match else None

    return reasoning, answer


def process_vision_info(messages):
    images = []
    videos = []
    for msg in messages:
        for content in msg["content"]:
            if content["type"] == "image":
                images.append(content["image"])
            elif content["type"] == "video":
                videos.append(content["video"])
    return images, videos


def format_prompt(example, task, style="alternative"):
    if style == "alternative":
        if task == "matching":
            return (
                f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n"
                f"I will provide you with five captions, one of which matches the cartoon image, "
                f"and the others are unrelated. Choose the caption that best matches the cartoon image: \n"
                f"A) {example['caption_choices'][0]}\n"
                f"B) {example['caption_choices'][1]}\n"
                f"C) {example['caption_choices'][2]}\n"
                f"D) {example['caption_choices'][3]}\n"
                f"E) {example['caption_choices'][4]}\n"
                f"Answer with the letter A, B, C, D, or E only."
            )
        elif task == "ranking":
            return (
                f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n",
                f"I will provide you with two captions; one of them is deemed funnier by people.",
                f"Which of the following captions is funnier: A: {example['caption_choices'][0]} B: {example['caption_choices'][1]}?\n",
                f"A) {example['caption_choices'][0]}\n",
                f"B) {example['caption_choices'][1]}\n\n",
                f"Think step by step.",
                f"Please write your reasoning between <think> </think> tags,",
                f"and your final answer between <answer> </answer> tags.\n",
                f"Your answer should look like: <think> your reasoning here </think> <answer> your answer here </answer>"
            )
        elif task == "explanation_from_pixels":
            return (
                f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n"
                f"The caption for the cartoon is: {example['caption_choices']}.\n"
                f"Explain why this caption is humorous in the context of the cartoon."
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        
    elif style == "better":
        if task == "matching":
            return (
                    f"In this task, you will see a description of an uncanny situation. Then, you will see five jokes — "
                    f"only one of which was written about the described situation. Pick which of the five choices truly corresponds to the described scene.\n\n"
                    f"###\n\n"
                    f"The image takes place in the following location: {example['image_location']}. "
                    f"{example['image_description']} {example['image_uncanny_description']}\n\n"
                    f"The scene includes: {', '.join([entity[30:] for entity in example['entities']])}.\n\n"
                    f"One of the following funny captions is most relevant to the scene:\n\n"
                    f"A) {example['caption_choices'][0]}\n"
                    f"B) {example['caption_choices'][1]}\n"
                    f"C) {example['caption_choices'][2]}\n"
                    f"D) {example['caption_choices'][3]}\n"
                    f"E) {example['caption_choices'][4]}\n\n"
                    f"The funny caption that matches the scene is:"
                )
        elif task == "ranking":
            return ( 
                    f"In this task, you will see a description of an uncanny situation. Then, you will see two jokes that were written about the situation. "
                    f"One of the jokes is better than the other one. Pick which of the two jokes is the one rated as funnier by people.\n\n"
                    f"###\n\n"
                    f"The image takes place in the following location: {example['image_location']}. "
                    f"{example['image_description']} {example['image_uncanny_description']}\n\n"
                    f"The scene includes: {', '.join([entity[30:] for entity in example['entities']])}. choices:\n\n"
                    f"A) {example['caption_choices'][0]}\n"
                    f"B) {example['caption_choices'][1]}\n\n"
                    f"The funnier is:"
                )
        elif task == "explanation_from_pixels":
            return (
                    f"The image is a cartoon from the New Yorker Cartoon Caption Contest. "
                    f"The caption for the cartoon is: {example['caption_choices']}. "
                    f"Explain why this caption is humorous in the context of the cartoon."
            )

        else:
            raise ValueError(f"Unknown task: {task}")

    else:
        raise ValueError(f"Prompt style '{style}' not implemented.")


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    from transformers import Qwen2_5_VLForConditionalGeneration  # specific import for Qwen2.5-VL
    from transformers import Qwen2VLForConditionalGeneration  # specific import for Qwen2-VL
    
    # Load VL model for Qwen2.5
    if args.model_name_or_path=="Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            device_map=device,
            torch_dtype="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        model.eval()
      
    
    # Load VL model for Qwen2
    elif args.model_name_or_path=="Qwen/Qwen2-VL-7B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            device_map=device,
            torch_dtype="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        model.eval()
    
    # Load other models
    else: 
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = args.model_name_or_path,
            load_in_4bit = True if "4bit" in args.model_name_or_path else False,
            dtype = None,  # Will auto-select based on GPU support
            
        )
        processor = tokenizer
        FastVisionModel.for_inference(model)
        """
        if "4bit" in args.model_name_or_path: 
            from unsloth import FastVisionModel
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name = args.model_name_or_path[5:],
                # This tells Unsloth where your fine-tuned LoRA weights are:
                adapter_name = args.model_name_or_path,
                load_in_4bit = True if "4bit" in args.model_name_or_path else False,
                dtype = None,  # Will auto-select based on GPU support
                
            )
            processor = tokenizer
            FastVisionModel.for_inference(model)
        else:
            from transformers import AutoModelForCausalLM
            # Qwen2VLForConditionalGeneration
            # Qwen2_5_VLForConditionalGeneration
            # AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, device_map=device, torch_dtype="auto"
            )
            processor = AutoProcessor.from_pretrained(args.model_name_or_path)
            model.eval()
        """
        


    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, name=args.config_name, split=args.split)
    else:
        dataset = load_dataset("json", data_files=args.dataset_path)["test"]

    results = []

    for example in tqdm(dataset, desc="Evaluating"):
        task = args.task_type
        image = example["image"]
        prompt = format_prompt(example, task, style=args.prompt_style)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(text=[text_input], images=image_inputs if image_inputs else None, videos=video_inputs if video_inputs else None, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=800, do_sample=False, repetition_penalty=1.2)
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        r, a = extract_reasoning_and_answer(prediction)

        results.append({
            "contest_number": example.get("contest_number", None),
            "task": task,
            "ground_truth": example.get("label", None),
            "prediction": prediction,
            "answer": a
        })

    output_path = f"./preds/{args.model_name_or_path}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved predictions to {output_path}")

    acc, nones = calculate_accuracy_alternative(results)
    print(f"Accuracy for ranking: {acc}")
    print(f"None for ranking / total: {nones}, {len(results)}")


def calculate_accuracy(data_task):
    accuracy = 0
    nones = 0
    for i in range(len(data_task)):

        prediction = data_task[i]['prediction']
        # Use regex to extract the letter before ')'
        r, a = extract_reasoning_and_answer(prediction)
        if (a != None) :
            choice = a
            if(data_task[i]['ground_truth'] == choice):
                accuracy += 1
        else:
            nones += 1
    
    return accuracy / len(data_task) * 100, nones

def calculate_accuracy_alternative(data_task):
    accuracy = 0
    nones = 0
    for i in range(len(data_task)):

        prediction = data_task[i]['prediction']
        r, a = extract_reasoning_and_answer(prediction)
        if a is not None:
            if(len(a) == 1):
                choice = a
                if(data_task[i]['ground_truth'].upper() == choice.upper()):
                    accuracy += 1
            else:

                match = re.search(r'([A-Z])\)', a.upper())
                if (match != None) :
                    choice = match.group(1)
                    if(data_task[i]['ground_truth'].upper() == choice.upper()):
                        accuracy += 1
                else:
                    nones += 1
        else:
            nones += 1
    
    return accuracy / len(data_task) * 100, nones

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Model Evaluator")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HuggingFace model name or path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_name", type=str, help="HuggingFace dataset name")
    group.add_argument("--dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument("--config_name", type=str, required=False, help="Config name for multi-config datasets")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, test, validation)")
    parser.add_argument("--task_type", type=str, choices=["matching", "ranking", "explanation_from_pixels"], default="ranking", help="Optional task filter")
    parser.add_argument("--prompt_style", type=str, default="alternative", help="Prompt style to use")
    args = parser.parse_args()

    run_inference(args)
