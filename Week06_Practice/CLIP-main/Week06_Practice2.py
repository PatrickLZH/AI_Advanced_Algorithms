import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path

class CLIPImageRetrieval:
    def __init__(self, model_name="ViT-B/32"):
        """
        初始化CLIP图像检索系统
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.image_embeddings = []
        self.image_paths = []
        
    def extract_image_features(self, image_path):
        """
        提取单张图像的特征向量
        """
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # 归一化特征向量
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    
    def build_image_index(self, image_folder):
        """
        构建图像索引库
        """
        self.image_embeddings = []
        self.image_paths = []
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 遍历文件夹中的所有图像
        for file_path in Path(image_folder).iterdir():
            if file_path.suffix.lower() in image_extensions:
                try:
                    features = self.extract_image_features(file_path)
                    self.image_embeddings.append(features.flatten())
                    self.image_paths.append(str(file_path))
                    print(f"已处理图像: {file_path.name}")
                except Exception as e:
                    print(f"处理图像 {file_path.name} 时出错: {e}")
        
        self.image_embeddings = np.array(self.image_embeddings)
        print(f"索引库构建完成，共包含 {len(self.image_paths)} 张图像")
    
    def search(self, query_image_path, top_k=5):
        """
        在索引库中搜索与查询图像最相似的图像
        """
        # 提取查询图像特征
        query_features = self.extract_image_features(query_image_path)
        query_features = query_features.flatten().reshape(1, -1)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_features, self.image_embeddings)[0]
        
        # 获取Top-K最相似的图像
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.image_paths[idx],
                'similarity': similarities[idx]
            })
            
        return results
    
    def visualize_results(self, query_image_path, results):
        """
        可视化检索结果
        """
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 3))
        
        # 显示查询图像
        query_image = Image.open(query_image_path)
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')
        
        # 显示检索结果
        for i, result in enumerate(results):
            result_image = Image.open(result['image_path'])
            axes[i+1].imshow(result_image)
            axes[i+1].set_title(f"Top {i+1}\nSim: {result['similarity']:.4f}")
            axes[i+1].axis('off')
            
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    print("CLIP图像检索系统已准备就绪")
    print("请按以下方式使用：")
    print("1. 调用 build_image_index() 构建图像索引库")
    print("2. 调用 search() 进行图像检索")
    print("3. 调用 visualize_results() 可视化检索结果")
    
    # 初始化检索系统
    retrieval_system = CLIPImageRetrieval()
    
    # 构建图像索引库（假设图像存储在'database_images'文件夹中）
    retrieval_system.build_image_index("database_images")
    
    # 执行图像检索（示例）
    # 查询图像
    query_image_path = "./database_images/42.png"
    results = retrieval_system.search(query_image_path, top_k=5)
    # 打印结果
    for i, result in enumerate(results):
        print(f"Top {i+1}: {result['image_path']} (相似度: {result['similarity']:.4f})")
    
    # 可视化检索结果
    retrieval_system.visualize_results(query_image_path, results)
    
