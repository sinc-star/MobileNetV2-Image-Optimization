import os
import sys
import sqlite3
import requests
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class UnsplashDatasetDownloader:
    def __init__(self, db_path, output_dir='data/unsplash/images', max_workers=10):
        """
        Unsplash数据集预下载器
        
        Args:
            db_path (str): 数据库文件路径
            output_dir (str): 图片保存目录
            max_workers (int): 并发下载线程数
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_image_urls(self):
        """
        从数据库加载所有图片URL
        
        Returns:
            list: [(photo_id, image_url), ...]
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT photo_id, photo_image_url FROM photos")
            urls = cursor.fetchall()
            conn.close()
            return urls
        except Exception as e:
            print(f"加载数据库失败: {e}")
            return []
    
    def download_image(self, photo_id, image_url):
        """
        下载单张图片
        
        Args:
            photo_id (str): 图片ID
            image_url (str): 图片URL
        
        Returns:
            tuple: (photo_id, success, error_message)
        """
        try:
            # 确保URL格式正确
            if not image_url.startswith(('http://', 'https://')):
                if image_url.startswith('images.unsplash.com'):
                    image_url = f"https://{image_url}"
                else:
                    return (photo_id, False, "Invalid URL format")
            
            # 添加尺寸参数
            if '?' in image_url:
                resized_url = f"{image_url}&w=224&h=224"
            else:
                resized_url = f"{image_url}?w=224&h=224"
            
            # 下载图片
            response = requests.get(resized_url, timeout=30)
            if response.status_code != 200:
                return (photo_id, False, f"HTTP {response.status_code}")
            
            # 解码图片
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return (photo_id, False, "Failed to decode image")
            
            # 保存图片
            output_path = os.path.join(self.output_dir, f"{photo_id}.jpg")
            cv2.imwrite(output_path, image)
            
            return (photo_id, True, None)
            
        except requests.exceptions.Timeout:
            return (photo_id, False, "Timeout")
        except Exception as e:
            return (photo_id, False, str(e))
    
    def download_all(self):
        """
        下载所有图片
        
        Returns:
            dict: 下载统计信息
        """
        print(f"从数据库加载图片URL...")
        urls = self.load_image_urls()
        
        if not urls:
            print("数据库中没有找到图片URL")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"找到 {len(urls)} 张图片")
        print(f"开始下载到 {self.output_dir}")
        print(f"并发线程数: {self.max_workers}")
        print()
        
        # 统计信息
        stats = {
            'total': len(urls),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        # 并发下载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_image, photo_id, image_url): (photo_id, image_url)
                for photo_id, image_url in urls
            }
            
            # 使用进度条
            with tqdm(total=len(urls), desc="下载进度") as pbar:
                for future in as_completed(futures):
                    photo_id, success, error = future.result()
                    
                    if success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                        stats['errors'].append((photo_id, error))
                    
                    pbar.update(1)
        
        # 打印统计信息
        print(f"\n下载完成!")
        print(f"总计: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        
        if stats['errors']:
            print(f"\n失败图片 (前10个):")
            for photo_id, error in stats['errors'][:10]:
                print(f"  {photo_id}: {error}")
        
        return stats

def main():
    """
    主函数
    """
    # 配置
    db_path = 'data/unsplash/db/unsplash.db'
    output_dir = 'data/unsplash/images'
    max_workers = 10  # 并发线程数，根据网络情况调整
    
    print("=" * 60)
    print("Unsplash 数据集预下载工具")
    print("=" * 60)
    print()
    
    # 检查数据库
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return
    
    # 创建下载器
    downloader = UnsplashDatasetDownloader(
        db_path=db_path,
        output_dir=output_dir,
        max_workers=max_workers
    )
    
    # 下载所有图片
    stats = downloader.download_all()
    
    print()
    print("=" * 60)
    print(f"图片已保存到: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
