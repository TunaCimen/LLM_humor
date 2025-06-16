# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hf_xet
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from peft import LoraConfig


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, contest_number, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, contest in zip(contents, solution, contest_number):
        parts = content.split("assistant", 1)
        if len(parts) == 2:
            content = parts[1]
        else:
            # fallback if "assistant" isn’t found
            content = parts[0]
        reward = 0.0
        student_answer = ""
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                #sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                #ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                ground_truth = sol.strip()
                
                # Extract answer from content if it has think/answer tags
                #content_match = re.search(r'<answer>(.*?)</answer>', content)
                content_match = re.search(
                    r"<answer>\s*([AB])\s*</answer>",
                    content,
                    flags=re.DOTALL,  # DOTALL makes `.` match `\n`
                )
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                #print(f"Predicted: {student_answer}, Ground Truth: {ground_truth}")
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Contest number: {contest}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer (ground truth): {sol}\n")
                    f.write(f"Predicted: {student_answer}, Ground Truth: {ground_truth}")
                    
        rewards.append(reward)
        
    
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    #pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    matches = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for content in completion_contents:
        parts = content.split("assistant", 1)
        if len(parts) == 2:
            content = parts[1].strip()
        else:
            # fallback if "assistant" isn’t found
            content = parts[0].strip()

        matches.append(re.fullmatch(pattern, content, re.DOTALL))
    #matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.rename_column("label", "solution")
    dataset = dataset.shuffle(seed=42)



    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }


    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (A or B) in <answer> </answer> tags."

    def make_conversation_image(example):
        problem = (
                f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n",
                f"I will provide you with two captions; one of them is deemed funnier by people.",
                f"Which of the following captions is funnier: A: {example['caption_choices'][0]} B: {example['caption_choices'][1]}?\n",
                f"A) {example['caption_choices'][0]}\n",
                f"B) {example['caption_choices'][1]}\n\n",
            )
        
        problem_ = (
                f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n",
                f"I will provide you with two captions; one of them is deemed funnier by people.",
                f"Think step by step:\n",
                f"1- Understand the visual setting and what makes it strange or surprising.\n",
                f"2- Identify who is most likely speaking in the cartoon.\n",
                f"3- Come up with a plausible story or dynamic behind the scene.\n",
                f"4- Analyze the humor, metaphors, and wordplay in the given captions.\n",
                f"Finally, decide which caption is funnier, and explain your reasoning.\n",
            )
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem)},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Qwen2VL’s projection layers
        lora_dropout=0.1,
        bias="none",
    )  
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=lora_cfg,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
