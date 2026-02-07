import sqlite3
import csv
import os

def create_database(db_path, csv_file):
    """
    创建数据库并导入元数据
    
    Args:
        db_path (str): 数据库文件路径
        csv_file (str): CSV文件路径
    """
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id TEXT UNIQUE,
        photo_image_url TEXT,
        photo_width INTEGER,
        photo_height INTEGER,
        exif_camera_make TEXT,
        exif_camera_model TEXT,
        exif_iso INTEGER,
        exif_aperture_value REAL,
        exif_focal_length REAL,
        exif_exposure_time TEXT,
        ai_description TEXT
    )
    ''')
    
    # 导入数据
    with open(csv_file, 'r', encoding='utf-8') as f:
        # 读取表头
        header = f.readline().strip().split('\t')
        photo_id_idx = header.index('photo_id')
        image_url_idx = header.index('photo_image_url')
        width_idx = header.index('photo_width')
        height_idx = header.index('photo_height')
        
        # 检查是否有EXIF字段
        exif_make_idx = header.index('exif_camera_make') if 'exif_camera_make' in header else -1
        exif_model_idx = header.index('exif_camera_model') if 'exif_camera_model' in header else -1
        exif_iso_idx = header.index('exif_iso') if 'exif_iso' in header else -1
        exif_aperture_idx = header.index('exif_aperture_value') if 'exif_aperture_value' in header else -1
        exif_focal_idx = header.index('exif_focal_length') if 'exif_focal_length' in header else -1
        exif_exposure_idx = header.index('exif_exposure_time') if 'exif_exposure_time' in header else -1
        ai_desc_idx = header.index('ai_description') if 'ai_description' in header else -1
        
        # 插入数据
        count = 0
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(photo_id_idx, image_url_idx):
                photo_id = parts[photo_id_idx]
                image_url = parts[image_url_idx]
                
                # 获取其他字段
                width = parts[width_idx] if width_idx < len(parts) else None
                height = parts[height_idx] if height_idx < len(parts) else None
                exif_make = parts[exif_make_idx] if exif_make_idx >= 0 and exif_make_idx < len(parts) else None
                exif_model = parts[exif_model_idx] if exif_model_idx >= 0 and exif_model_idx < len(parts) else None
                exif_iso = parts[exif_iso_idx] if exif_iso_idx >= 0 and exif_iso_idx < len(parts) else None
                exif_aperture = parts[exif_aperture_idx] if exif_aperture_idx >= 0 and exif_aperture_idx < len(parts) else None
                exif_focal = parts[exif_focal_idx] if exif_focal_idx >= 0 and exif_focal_idx < len(parts) else None
                exif_exposure = parts[exif_exposure_idx] if exif_exposure_idx >= 0 and exif_exposure_idx < len(parts) else None
                ai_desc = parts[ai_desc_idx] if ai_desc_idx >= 0 and ai_desc_idx < len(parts) else None
                
                # 插入数据
                cursor.execute('''
                INSERT OR IGNORE INTO photos 
                (photo_id, photo_image_url, photo_width, photo_height, 
                 exif_camera_make, exif_camera_model, exif_iso, 
                 exif_aperture_value, exif_focal_length, exif_exposure_time, 
                 ai_description) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (photo_id, image_url, width, height, 
                      exif_make, exif_model, exif_iso, 
                      exif_aperture, exif_focal, exif_exposure, 
                      ai_desc))
                
                count += 1
                if count % 1000 == 0:
                    print(f'Inserted {count} records')
    
    # 提交并关闭
    conn.commit()
    conn.close()
    print(f'Created database with {count} records')

if __name__ == '__main__':
    # 创建数据库目录
    db_dir = 'data/unsplash/db'
    os.makedirs(db_dir, exist_ok=True)
    
    # 数据库路径
    db_path = os.path.join(db_dir, 'unsplash.db')
    
    # CSV文件路径
    csv_file = 'data/upsplash/photos.csv000'
    
    # 创建数据库
    create_database(db_path, csv_file)