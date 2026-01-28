import os
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# 1. 环境准备
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def test(m, p):
    inputs = tokenizer(p, return_tensors="pt").to(m.device)
    with torch.no_grad():
        outputs = m.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 2. 算子镜像实验
def reverse_operator_logic(model):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        embeds = new_model.get_input_embeddings().weight
        
        # 获取 + 和 - 的 ID
        plus_id = tokenizer.encode("+", add_special_tokens=False)[0]
        minus_id = tokenizer.encode("-", add_special_tokens=False)[0]
        plus_cn_id = tokenizer.encode("加", add_special_tokens=False)[0]
        minus_cn_id = tokenizer.encode("减", add_special_tokens=False)[0]
        
        # 交换它们的几何身份 (交换算子坐标)
        # 这样不破坏输出层，只是让模型“认错”操作符
        plus_vec = embeds[plus_id].clone()
        minus_vec = embeds[minus_id].clone()
        embeds[plus_id] = minus_vec
        embeds[minus_id] = plus_vec
        
        # 同时交换中文算子
        plus_cn_vec = embeds[plus_cn_id].clone()
        minus_cn_vec = embeds[minus_cn_id].clone()
        embeds[plus_cn_id] = minus_cn_vec
        embeds[minus_cn_id] = plus_cn_vec
        
    return new_model

# 3. 执行“底层协议篡改”
flipped_model = reverse_operator_logic(model)

print("--- 算子反转实验 ---")
print("正常模型 5+1 =", test(model, "5+1="))
print("篡改模型 5+1 =", test(flipped_model, "5+1=")) # 预期结果：4
print("篡改模型 5-1 =", test(flipped_model, "5-1=")) # 预期结果：6