# DeepSeek-R1-Medical-COT

Fine-tuned version of DeepSeek-R1-Distill-Llama-8B optimized for medical question answering with complex Chain of Thought (CoT) reasoning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/DeepSeek-R1-Medical-COT/blob/main/DeepSeek_R1_Medical_CoT_Finetuning.ipynb)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-kingabzpro/DeepSeek--R1--Medical--COT-yellow)](https://huggingface.co/kingabzpro/DeepSeek-R1-Medical-COT)
[![wandb](https://img.shields.io/badge/Weights%20&%20Biases-Experiment-blue)](https://wandb.ai/username/Fine-tune-DeepSeek-R1)

## Model Description

This repository contains a fine-tuned version of the [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B) model, specifically optimized for medical question answering with step-by-step reasoning. The model has been trained to provide detailed Chain of Thought (CoT) responses to complex medical questions.

### Key Features:
- Built on DeepSeek-R1-Distill-Llama-8B (8B parameters)
- Fine-tuned specifically for medical domain questions
- Generates step-by-step reasoning before providing final answers
- Optimized with Unsloth for efficient training and inference

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

# Load the model and tokenizer
model_name = "kingabzpro/DeepSeek-R1-Medical-COT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For faster inference with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 4096,  # Adjust as needed
    dtype = "bfloat16",     # Use "float16" if bfloat16 is not supported
    load_in_4bit = True     # For memory efficiency
)

# Enable optimized inference mode
FastLanguageModel.for_inference(model)

# Define your medical question
question = "A 60-year-old man with a history of smoking presents with persistent cough, hemoptysis, and unintentional weight loss. Imaging reveals a lung mass, and laboratory tests show hypercalcemia. What is the most likely underlying cause of his hypercalcemia, and what is the probable diagnosis?"

# Format the question using the structured prompt
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
{0}

### Response:
{1}"""

inputs = tokenizer([prompt_style.format(question, "")], return_tensors = "pt").to("cuda")

# Generate the response
outputs = model.generate(
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 1200,
    use_cache = True
)

# Display the response
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
```

## Training Details

The model was fine-tuned using the following approach:

1. **Base Model**: [unsloth/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)
2. **Training Method**: QLoRA (Quantized Low-Rank Adaptation)
3. **LoRA Configuration**:
   - Rank: 16
   - Alpha: 16
   - Target modules: query, key, value, output, gate, up and down projections
4. **Training Parameters**:
   - Batch size: 1 per device
   - Gradient accumulation steps: 8
   - Learning rate: 2e-4 with linear scheduler
   - Weight decay: 0.01
   - Training steps: 60 with 5 warmup steps
   - Optimizer: AdamW 8-bit

## Dataset

The model was trained on a curated dataset of medical questions paired with detailed Chain of Thought reasoning and final responses. The dataset formatting uses a specific prompt template that encourages step-by-step reasoning.

## Limitations

- The model is specialized for medical question answering and may not perform optimally on other domains
- As with all medical AI systems, this model should be used as a tool to assist healthcare professionals, not as a replacement for professional medical advice
- The model may occasionally generate incorrect or incomplete information

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{deepseek-r1-medical-cot,
  author = {Sayiqa},
  title = {DeepSeek-R1-Medical-COT: A Fine-tuned Medical Question Answering Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Sayiqajabeen/DeepSeek-R1-Medical-COT}}
}
```

## License

This model inherits the license of the base DeepSeek-R1-Distill-Llama-8B model. Please refer to the [original model's license](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B) for details.

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for their efficient fine-tuning framework
- [DeepSeek-AI](https://github.com/deepseek-ai) for the base model
- [Hugging Face](https://huggingface.co/) for hosting the model
- [Weights & Biases](https://wandb.ai/) for experiment tracking
