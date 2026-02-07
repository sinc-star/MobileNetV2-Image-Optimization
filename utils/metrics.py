import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(predictions, targets):
    """
    计算评估指标
    
    Args:
        predictions (np.ndarray): 预测值
        targets (np.ndarray): 真实值
    
    Returns:
        dict: 评估指标
    """
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # 计算MAE
    mae = mean_absolute_error(targets, predictions)
    
    # 计算R²
    r2 = r2_score(targets, predictions)
    
    # 计算每个参数的指标
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'param_metrics': {}
    }
    
    # 两参数版本
    if predictions.shape[1] == 2:
        metrics['param_metrics']['exposure'] = {
            'rmse': np.sqrt(mean_squared_error(targets[:, 0], predictions[:, 0])),
            'mae': mean_absolute_error(targets[:, 0], predictions[:, 0]),
            'r2': r2_score(targets[:, 0], predictions[:, 0])
        }
        metrics['param_metrics']['saturation'] = {
            'rmse': np.sqrt(mean_squared_error(targets[:, 1], predictions[:, 1])),
            'mae': mean_absolute_error(targets[:, 1], predictions[:, 1]),
            'r2': r2_score(targets[:, 1], predictions[:, 1])
        }
    # 五参数版本
    elif predictions.shape[1] == 5:
        param_names = ['exposure', 'contrast', 'saturation', 'highlight', 'shadow']
        for i, param_name in enumerate(param_names):
            metrics['param_metrics'][param_name] = {
                'rmse': np.sqrt(mean_squared_error(targets[:, i], predictions[:, i])),
                'mae': mean_absolute_error(targets[:, i], predictions[:, i]),
                'r2': r2_score(targets[:, i], predictions[:, i])
            }
    
    return metrics

def print_metrics(metrics):
    """
    打印指标
    
    Args:
        metrics (dict): 评估指标
    """
    print('Overall Metrics:')
    print(f'RMSE: {metrics["rmse"]:.4f}')
    print(f'MAE: {metrics["mae"]:.4f}')
    print(f'R2: {metrics["r2"]:.4f}')
    
    if 'param_metrics' in metrics:
        print('\nPer Parameter Metrics:')
        for param_name, param_metrics in metrics['param_metrics'].items():
            print(f'\n{param_name}:')
            print(f'  RMSE: {param_metrics["rmse"]:.4f}')
            print(f'  MAE: {param_metrics["mae"]:.4f}')
            print(f'  R2: {param_metrics["r2"]:.4f}')