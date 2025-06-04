import torch
import json
import random
import argparse
from PIL import Image
from trl import SFTTrainer, SFTConfig

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, default_data_collator, AutoTokenizer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import FastVisionModel # FastLanguageModel for LLMs


random.seed(42)
def build_prompt_for_caption_pair(caption_a_text, caption_b_text):
    prompt_lines = [
        f"The image is a cartoon from the New Yorker Cartoon Caption Contest.\n",
        f"I will provide you with two captions; one of them is deemed funnier by people.",
        f"Which of the following captions is funnier: A: {caption_a_text} B: {caption_b_text}?\n",
        f"A) {caption_a_text}\n",
        f"B) {caption_b_text}\n\n",
        f"Think step by step.",
        f"Please write your reasoning between <think> </think> tags,",
        f"and your final answer between <answer> </answer> tags.\n",
        f"Your answer should look like: <think> your reasoning here </think> <answer> your answer here </answer>"
    ]
    return "\n".join(prompt_lines)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL model with SFT.")
    parser.add_argument("--model_size", choices=["7b", "72b"], default="7b", help="Model size: 7b or 72b.")
    parser.add_argument("--dataset_path", type=str, default="dataset.json", help="Path to JSON dataset.")
    return parser.parse_args()

def load_qwen2vl(model_size, device_map=None):
    if model_size == "7b":
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
    else:
        model_id = "Qwen/Qwen2-VL-72B-Instruct"

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    return model, tokenizer



def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# { "image": "/path/to/somefile.jpg"}
def load_pil_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img


def convert_to_conversation(sample):
    caption_a_text = sample['caption_choices']['0'][0][0]
    caption_b_text = sample['caption_choices']['0'][0][1]
    instruction = build_prompt_for_caption_pair(caption_a_text, caption_b_text)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": load_pil_image(sample["image"])}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["response"]}
            ]
        }
    ]
    return {"messages": conversation}


def main():
    args = parse_args()
    model, tokenizer = load_qwen2vl(args.model_size)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )


    dataset = load_data(args.dataset_path)
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    val_dataset = random.sample(converted_dataset, 200)
    FastVisionModel.for_training(model)
    
    trainer_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=500,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
        run_name=f"qwen2vl_sft_{args.model_size}",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
        gradient_checkpointing_kwargs={'use_reentrant':False}
    )



    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        eval_dataset=val_dataset,
        args=trainer_args
    )

    trainer.train()

    if True: 
        model.save_pretrained_merged("unsloth_sft", tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()
