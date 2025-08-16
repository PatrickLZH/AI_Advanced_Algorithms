from transformers import AutoTokenizer, OPTForCausalLM
import torch
import numpy as np

class OPTTextGenerator:
    def __init__(self, model_name="facebook/opt-350m"):
        """
        初始化OPT文本生成器
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = OPTForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
        """
        使用OPT模型生成文本
        """
        # 编码输入提示
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def next_token_prediction_demo(self, prompt):
        """
        演示Next Token Prediction原理
        """
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 获取模型输出（logits）
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits
        
        # 获取最后一个token的logits
        next_token_logits = logits[0, -1, :]
        
        # 应用softmax获取概率分布
        probabilities = torch.softmax(next_token_logits, dim=-1)
        
        # 获取概率最高的几个token
        top_k = 10
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # 解码top tokens
        top_tokens = []
        for i in range(top_k):
            token = self.tokenizer.decode([top_indices[i].item()])
            prob = top_probs[i].item()
            top_tokens.append((token, prob))
        
        return top_tokens
    
    def step_by_step_generation(self, prompt, steps=5):
        """
        逐步生成文本，展示每一步的next token prediction
        """
        current_text = prompt
        generation_steps = []
        
        for step in range(steps):
            # 获取下一步可能的tokens
            next_tokens = self.next_token_prediction_demo(current_text)
            
            # 选择概率最高的token（贪婪搜索）
            next_token = next_tokens[0][0]
            next_token_prob = next_tokens[0][1]
            
            # 更新文本
            current_text += next_token
            
            # 记录步骤信息
            generation_steps.append({
                'step': step + 1,
                'current_text': current_text,
                'predicted_token': next_token,
                'probability': next_token_prob,
                'top_candidates': next_tokens[:5]
            })
            
            # 如果生成了结束符，则停止
            if next_token == self.tokenizer.eos_token:
                break
                
        return generation_steps

def explain_next_token_prediction():
    """
    解释Next Token Prediction原理
    """
    print("Next Token Prediction 原理解释")
    print("=" * 50)
    
    explanation = """
    1. 模型架构:
       - OPT是基于Transformer的自回归语言模型
       - 使用因果注意力机制(只关注前面的token)
       - 通过多层注意力和前馈网络处理序列
    
    2. 预测过程:
       - 输入序列: x₁, x₂, ..., xₜ
       - 模型输出: 每个位置对应词汇表上所有token的概率分布
       - 下一token预测: P(xₜ₊₁ | x₁, x₂, ..., xₜ)
    
    3. 概率计算:
       - logits = 模型最后一层输出
       - probabilities = softmax(logits)
       - 选择概率最高的token或按概率采样
    
    4. 生成策略:
       - 贪婪搜索: 每次选择概率最高的token
       - 随机采样: 按概率分布随机选择
       - Top-k采样: 从概率最高的k个token中采样
       - Top-p采样: 从累积概率超过p的token中采样
    """
    
    print(explanation)


def compare_generation_strategies():
    """
    对比不同文本生成策略
    """
    generator = OPTTextGenerator("facebook/opt-125m")
    
    prompt = "Artificial intelligence"
    
    print("不同生成策略对比")
    print("=" * 60)
    print(f"输入提示: {prompt}")
    print()
    
    # 1. 贪婪搜索
    greedy_output = generator.generate_text(
        prompt, 
        max_length=80, 
        temperature=1.0,
        top_k=1,  # 贪婪搜索
        top_p=0.0
    )
    print("1. 贪婪搜索:")
    print(f"   {greedy_output}")
    print()
    
    # 2. 随机采样 (高温度)
    random_high_temp = generator.generate_text(
        prompt,
        max_length=80,
        temperature=1.5,
        top_k=0,
        top_p=0.0
    )
    print("2. 随机采样 (高温度=1.5):")
    print(f"   {random_high_temp}")
    print()
    
    # 3. 随机采样 (低温度)
    random_low_temp = generator.generate_text(
        prompt,
        max_length=80,
        temperature=0.5,
        top_k=0,
        top_p=0.0
    )
    print("3. 随机采样 (低温度=0.5):")
    print(f"   {random_low_temp}")
    print()
    
    # 4. Top-k采样
    top_k_sampling = generator.generate_text(
        prompt,
        max_length=80,
        temperature=1.0,
        top_k=50,
        top_p=0.0
    )
    print("4. Top-k采样 (k=50):")
    print(f"   {top_k_sampling}")
    print()
    
    # 5. Top-p采样
    top_p_sampling = generator.generate_text(
        prompt,
        max_length=80,
        temperature=1.0,
        top_k=0,
        top_p=0.95
    )
    print("5. Top-p采样 (p=0.95):")
    print(f"   {top_p_sampling}")


# 使用示例
if __name__ == "__main__":
    # 初始化OPT生成器
    generator = OPTTextGenerator("facebook/opt-125m")  # 使用较小的模型以便快速演示
    
    # 示例1: 基本文本生成
    print("OPT文本生成实验")
    print("=" * 60)
    
    prompt = "The importance of artificial intelligence in modern society"
    print(f"输入提示: {prompt}")
    
    generated_text = generator.generate_text(prompt, max_length=150, temperature=0.8)
    print(f"生成文本: {generated_text}")
    print()
    
    # 示例2: Next Token Prediction演示
    print("Next Token Prediction原理演示")
    print("=" * 60)
    
    short_prompt = "The weather today is"
    print(f"输入文本: {short_prompt}")
    
    next_tokens = generator.next_token_prediction_demo(short_prompt)
    print("最可能的下一个token:")
    for i, (token, prob) in enumerate(next_tokens):
        print(f"  {i+1:2d}. '{token}' (概率: {prob:.4f})")
    print()
    
    # 示例3: 逐步生成过程
    print("逐步生成过程演示")
    print("=" * 60)
    
    step_prompt = "Once upon a time"
    print(f"初始文本: {step_prompt}")
    
    steps = generator.step_by_step_generation(step_prompt, steps=5)
    for step_info in steps:
        print(f"\n步骤 {step_info['step']}:")
        print(f"  当前文本: {step_info['current_text']}")
        print(f"  预测token: '{step_info['predicted_token']}' (概率: {step_info['probability']:.4f})")
        print(f"  候选tokens: {[f'{t[0]}({t[1]:.3f})' for t in step_info['top_candidates']]}")

    # 运行原理解释
    explain_next_token_prediction()

    # 运行策略对比
    compare_generation_strategies()