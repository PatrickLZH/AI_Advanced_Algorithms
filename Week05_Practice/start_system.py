#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接口版本启动脚本
使用接口解耦，不依赖服务调用
"""

import subprocess
import os

def main():
    """主函数"""
    print("🎯 倒排索引检索系统 - 接口解耦版本")
    print("=" * 50)
    
    # 1. 构建离线索引
    print("\n📦 步骤1: 构建离线索引")
    print("-" * 30)
    
    if not os.path.exists('models/index_data.json'):
        print("   索引文件不存在，开始构建...")
        try:
            subprocess.run(["python", "offline/offline_index.py"], check=True, cwd=os.path.dirname(__file__))
            print("✅ 离线索引构建完成")
        except subprocess.CalledProcessError:
            print("❌ 离线索引构建失败")
            return
    else:
        print("✅ 索引文件已存在，跳过构建")
    
    # 2. 启动UI界面
    print("\n🖥️  步骤2: 启动UI界面")
    print("-" * 30)
    
    try:
        print("🚀 启动UI界面...")
        print("   使用接口解耦方式，无需启动在线服务")
        subprocess.run(["python", "ui/portal.py"], cwd=os.path.dirname(__file__))
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断，正在退出...")
    except Exception as e:
        print(f"❌ 启动UI界面失败: {e}")

if __name__ == "__main__":
    main() 