import sys
import traceback

# Redirect stdout and stderr to a file
log_file = open('eagenerate_modeling.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

try:
    from eagle.modeling_eagle import EAGLE
    from fastchat.model import get_conversation_template
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Print Python version and environment info
    import platform
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load the base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    # Initialize the EAGLE model
    print("Initializing EAGLE model...")
    model = EAGLE(
        base_model=base_model,
        eagle_path='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
        use_tree_attn=True
    )

    # Prepare the input
    your_message = "Hello"
    print(f"Input message: '{your_message}'")

    # Create a conversation template and append your message
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.cuda()
    attention_mask = torch.ones_like(input_ids)

    # Generate output
    print("Generating response...")
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=0.5,
        max_new_tokens=512
    )

    # Decode the output IDs to get the generated text
    if isinstance(output_ids, list):
        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    print("\nGenerated response:")
    print(output)

    # Save the output to a file
    with open('output_modeling.txt', 'w') as f:
        f.write(output)

    print("Output saved to output_modeling.txt")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

finally:
    # Close the log file
    log_file.close()
    # Reset stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__