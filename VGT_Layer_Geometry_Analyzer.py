import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from transformers import AutoModelForCausalLM, AutoTokenizer
proxy = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
# --- 1. 环境与模型准备 ---
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# --- 2. 定义 Token 集 ---
digits = [str(i) for i in range(10)]
logic_anchors = ["因为", "所以", "因此", "那么"]
control_tokens = ["苹果", "天空", "跑步", "音乐"] 
all_tokens = digits + logic_anchors + control_tokens

# --- 3. 提取多层 Hidden States (健壮版) ---
# 为了避免分词歧义，我们逐个 Token 获取其在独立编码下的表征
# 这样能最纯粹地观察 VGT 预言的“静态”与“动态”变换
results_map = {0: [], 22: [], 23: []}
valid_token_labels = []

print("正在提取层特征...")
for t in all_tokens:
    # 统一增加前缀空格以模拟自然上下文，并取最后一个 token id
    # 很多模型对 "2" 和 " 2" 的编码是不一样的
    inputs = tokenizer(t, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 取序列的最后一个位置 (Last hidden state of the token)
        for l_idx in [0, 22, 23]:
            # hidden_states[layer][batch, seq, dim]
            feat = outputs.hidden_states[l_idx][0, -1, :].cpu().float().numpy()
            results_map[l_idx].append(feat)
    valid_token_labels.append(t)

# 转换为 numpy 数组 [N_tokens, Hidden_dim]
for l_idx in results_map:
    results_map[l_idx] = np.array(results_map[l_idx])

# --- 4. 几何分析函数 ---
def analyze_vgt_structure(target_vectors, layer_idx):
    # PCA 降维到 3D
    pca = PCA(n_components=3)
    coords = pca.fit_transform(target_vectors)
    
    d_coords = coords[:10]
    l_coords = coords[10:14]
    c_coords = coords[14:]
    
    # 计算数字步长与 CV
    steps = np.linalg.norm(np.diff(d_coords, axis=0), axis=1)
    cv = np.std(steps) / np.mean(steps)
    
    # 计算逻辑压缩比 (逻辑词内部平均距离 / 数字序列内部平均距离)
    l_dist = np.mean(pdist(l_coords))
    d_dist = np.mean(pdist(d_coords))
    compression = l_dist / d_dist
    
    return {
        "coords": coords,
        "d_pts": d_coords,
        "l_pts": l_coords,
        "c_pts": c_coords,
        "cv": cv,
        "compression": compression,
        "steps": steps
    }

# --- 5. 执行分析与可视化 ---
fig = plt.figure(figsize=(20, 7))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # 中文支持

final_stats = {}
layers_to_plot = [0, 22, 23]

for i, l_idx in enumerate(layers_to_plot):
    res = analyze_vgt_structure(results_map[l_idx], l_idx)
    final_stats[l_idx] = res
    
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    # 绘制数字流形线段
    ax.plot(res["d_pts"][:,0], res["d_pts"][:,1], res["d_pts"][:,2], 'b-o', label='数字(0-9)', alpha=0.5)
    # 绘制逻辑锚点
    ax.scatter(res["l_pts"][:,0], res["l_pts"][:,1], res["l_pts"][:,2], c='red', marker='X', s=120, label='逻辑锚点')
    # 绘制对照组
    ax.scatter(res["c_pts"][:,0], res["c_pts"][:,1], res["c_pts"][:,2], c='green', marker='^', s=60, label='对照组')
    
    # 标注逻辑词
    for j, txt in enumerate(logic_anchors):
        ax.text(res["l_pts"][j,0], res["l_pts"][j,1], res["l_pts"][j,2], txt, fontsize=9)

    ax.set_title(f"Layer {l_idx}\nCV={res['cv']:.3f} | Comp={res['compression']:.3f}x")
    ax.legend()

plt.tight_layout()
plt.show()

# --- 6. 判定报告 ---
print("\n" + "="*60)
print("【VGT 几何演化判定报告 - 修复版】")
print("="*60)
for l_idx in layers_to_plot:
    res = final_stats[l_idx]
    print(f"Layer {l_idx:02d}: CV={res['cv']:.3f}, 逻辑/数字压缩比={res['compression']:.4f}")
print("-" * 60)
print("结论：如果 Layer 23 的 CV 远大于 Layer 0，则证实了模型在深层进入了算法扭曲阶段。")
print("="*60)