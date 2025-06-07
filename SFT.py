import json
import random
import argparse
from PIL import Image
from unsloth import FastVisionModel
import torch
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


def build_prompt_for_caption_pair(caption_a_text, caption_b_text, contest_no):
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
    return "\n".join(prompt_lines)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with SFT.")
    parser.add_argument("--model_name", type=str, default=None, help="Model id.")
    parser.add_argument("--dataset_path", type=str, default="sft_data_50.json", help="Path to JSON dataset.")
    parser.add_argument("--is_4_bit", action="store_true", help="4-bit quantization.")
    parser.add_argument("--max_steps", type=int, default=250, help="Maximum steps for SFT.")


    return parser.parse_args()

def load_model(args, device_map=None):

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit = args.is_4_bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    return model, tokenizer


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_pil_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img


def convert_to_conversation(sample, i):
    caption_a_text = sample['caption_choices'][i][0][0]
    caption_b_text = sample['caption_choices'][i][0][1]
    answer = sample['caption_choices'][i][1]
    contest_number = sample["contest_number"]
    instruction = build_prompt_for_caption_pair(caption_a_text, caption_b_text, contest_number)

    response = sample["responses"][i]
    txt = f"<think>{response}</think> <answer>{answer}</answer>"

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
                {"type": "text", "text": txt}
            ]
        }
    ]
    return {"messages": conversation}


def main():
    args = parse_args()
    model, tokenizer = load_model(args)
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
    print(f"MODEL: {args.model_name}")
    print('Model loaded...')

    dataset = load_data(args.dataset_path)
    converted_dataset = []
    dataset_path = "./sft_data4.json" # MAY CHANGE
    dataset = load_data(dataset_path)
    for sample in dataset:
        for i in sample["responses"]:
            if i in sample['caption_choices']:
                converted_dataset.append(convert_to_conversation(sample, i))
    val_dataset = converted_dataset[:50]
    train_dataset = converted_dataset[50:]
    
    print('Dataset loaded...')
    

    FastVisionModel.for_training(model)
    
    trainer_args = SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
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
        run_name=f"sft_{args.model_name}",
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=trainer_args
    )
    
    print('Training starts...')
    trainer.train()


    """
    if "4bit" in args.model_name: 
        model.save_pretrained(f"sft_{args.model_name}")
        tokenizer.save_pretrained(f"sft_{args.model_name}")
    else:
        model.save_pretrained_merged(f"sft_{args.model_name}", tokenizer, save_method="merged_16bit")
    """
    
    
    model.save_pretrained(f"sft_{args.model_name}")
    tokenizer.save_pretrained(f"sft_{args.model_name}")
    
    

if __name__ == "__main__":
    main()
