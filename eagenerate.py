from eagle.model.ea_model import EaModel
from eagle.model.ea_trace import enable_ea_trace, disable_ea_trace
from fastchat.model import get_conversation_template
import torch
# Load the EAGLE model
model = EaModel.from_pretrained(
    base_model_path='meta-llama/Llama-3.1-8B-Instruct',
    ea_model_path='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
# Ensure the model is in evaluation mode
model.eval()

# Enable trace to see debug output
# print("Enabling EAGLE trace...")
# enable_ea_trace(only_first_loop=True)
your_message="Hello"
# Create a conversation template and append your message
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
# Tokenize the prompt and generate output
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()


output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
# Decode the output IDs to get the generated text
output=model.tokenizer.decode(output_ids[0])


print(output)
with open('output.txt', 'w') as f:
    f.write(output)
print("Output saved to output.txt")

# Disable trace when done
# disable_ea_trace()
