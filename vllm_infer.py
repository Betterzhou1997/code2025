import numpy as np
from vllm import LLM, SamplingParams


def generate(llm_, prompts_):
    results = []

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1000, frequency_penalty=0.2, presence_penalty=0, best_of=1,
        logprobs=1, include_stop_str_in_output=True,
        stop=None
    )
    outputs = llm_.generate(prompts_, sampling_params)
    for output in outputs:
        answer = output.outputs[0].text
        print(answer)
        # 安全处理logprobs
        logprobs = output.outputs[0].logprobs
        prob = []
        if logprobs and len(logprobs) > 0:
            first_token_logprobs = logprobs[0]
            prob = [
                (llm_.get_tokenizer().decode(k), np.exp(v.logprob))
                for k, v in first_token_logprobs.items()
            ]
        results.append({
            'answer': answer,
            'prob': prob
        })
    return results


if __name__ == '__main__':
    model_path = r"D:\model\Qwen\Qwen3-0___6B"
    tensor_parallel_size = 1
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True,
              dtype='auto',
              max_model_len=8192, enable_prefix_caching=True)
    print("============================================model loaded============================================")
    prompts = [f"你好，今天是星期{i}！" for i in range(1, 8)]
    res = generate(llm, prompts)


