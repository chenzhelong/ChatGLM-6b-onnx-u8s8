import re
import numpy as np
from tokenizer import ChatGLMTokenizer
# import torch
from onnxruntime import InferenceSession, SessionOptions


# Currently `MatMulInteger` and `DynamicQuantizeLinear` are only supported on CPU,
# although they are documented as supported on CUDA.
providers = ["CPUExecutionProvider"]

# if torch.cuda.is_available():
#     providers = ["CUDAExecutionProvider"] + providers


# Default paths
tokenizer_path = "chatglm-6b-int8-onnx-merged/sentencepiece.model"
onnx_model_path = "chatglm-6b-int8-onnx-merged/chatglm-6b-int8.onnx"


# input & output names
past_names = [f"past_{name}_{i}" for i in range(28) for name in ["key", "value"]]
present_names = [f"present_{name}_{i}" for i in range(28) for name in ["key", "value"]]
output_names = ["logits"] + present_names


# default kv_cache for first inference
default_past_key_values = {
    k: np.zeros((1, 0, 32, 128), dtype=np.float32) for k in past_names
}


def chat_template(history: list[tuple[str, str]], current: str):
    prompt = ""
    chat_round = 0
    for question, answer in history:
        prompt += f"[Round {chat_round}]\n问：{question}\n答：{answer}\n"
        chat_round += 1
    prompt += f"[Round {chat_round}]\n问：{current}\n答："
    return prompt


def process_response(response: str):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


class ChatGLMModel():

    def __init__(self, onnx_model_path=onnx_model_path, tokenizer_path=tokenizer_path, profile=False) -> None:
        self.tokenizer = ChatGLMTokenizer(tokenizer_path)
        options = SessionOptions()
        options.enable_profiling = profile
        self.session = InferenceSession(onnx_model_path, options, providers=providers)
        self.eop_token_id = self.tokenizer["<eop>"]


    def prepare_input(self, prompt: str):
        input_ids, prefix_mask = self.tokenizer.encode(prompt)

        input_ids = np.array([input_ids], dtype=np.longlong)
        prefix_mask = np.array([prefix_mask], dtype=np.longlong)

        return input_ids, prefix_mask, default_past_key_values


    def sample_next_token(self, logits: np.ndarray, top_k=50, top_p=0.7, temperature=1):
        # softmax with temperature
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / np.sum(exp_logits)

        # top k
        top_k_idx = np.argsort(-probs)[:top_k]
        top_k_probs = probs[top_k_idx]

        # top p
        cumsum_probs = np.cumsum(top_k_probs)
        top_k_probs[(cumsum_probs - top_k_probs) > top_p] = 0.0
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        # sample
        next_token = np.random.choice(top_k_idx, size=1, p=top_k_probs)
        return next_token[0].item()


    def generate_iterate(self, prompt: str, max_generated_tokens=100, top_k=50, top_p=0.7, temperature=1):
        input_ids, prefix_mask, past_key_values = self.prepare_input(prompt)
        output_tokens = []

        while True:
            inputs = {
                "input_ids": input_ids,
                "prefix_mask": prefix_mask,
                "use_past": np.array(len(output_tokens) > 0),
            }
            inputs.update(past_key_values)

            logits, *past_key_values = self.session.run(output_names, inputs)
            past_key_values = { k: v for k, v in zip(past_names, past_key_values) }

            next_token = self.sample_next_token(logits[0, -1], top_k=top_k, top_p=top_p, temperature=temperature)
            
            output_tokens += [next_token]

            if next_token == self.eop_token_id or len(output_tokens) > max_generated_tokens:
                break

            input_ids = np.array([[next_token]], dtype=np.longlong)
            prefix_mask = np.concatenate([prefix_mask, np.array([[0]], dtype=np.longlong)], axis=1)

            yield process_response(self.tokenizer.decode(output_tokens))

        return process_response(self.tokenizer.decode(output_tokens))

