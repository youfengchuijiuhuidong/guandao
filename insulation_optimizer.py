import itertools
import math
import numpy as np
from typing import List, Dict, Tuple

# 材料属性
MATERIALS = {
    '纳米气凝胶': {'导热系数': 0.02, '单价': 13274.3438, '单位': '立方米', '规格': '10mm', '容重': 207},
    '高温玻璃棉': {'导热系数': 0.04, '单价': 796.4615, '单位': '立方米', '规格': '50mm', '容重': 48},
    '硅酸铝针刺毯': {'导热系数': 0.09, '单价': 867.25, '单位': '立方米', '规格': '50mm', '容重': 128},
    '钠镁管保护层': {'导热系数': 0.05, '单价': 336.284, '单位': '米', '规格': 'Φ770x2/Φ790x2', '容重': 260},
    '钠镁硬质保温块': {'导热系数': 0.05, '单价': 256.638, '单位': '米', '规格': 'Φ790x70', '容重': 260}
}

class InsulationLayer:
    def __init__(self, material: str, thickness: float):
        self.material = material
        self.thickness = thickness  # mm
        self.properties = MATERIALS[material]
    
    def get_thermal_resistance(self, inner_diameter: float, length: float) -> float:
        """计算热阻 (K/W)"""
        outer_diameter = inner_diameter + 2 * self.thickness
        return math.log(outer_diameter / inner_diameter) / (2 * math.pi * self.properties['导热系数'] * length)
    
    def get_cost(self, inner_diameter: float, length: float) -> float:
        """计算成本 (元)"""
        outer_diameter = inner_diameter + 2 * self.thickness
        if self.properties['单位'] == '米':
            return self.properties['单价'] * length
        else:
            volume = math.pi * length * (outer_diameter**2 - inner_diameter**2) / 4 / 1000000  # 转换为立方米
            return self.properties['单价'] * volume

class InsulationScheme:
    def __init__(self, layers: List[InsulationLayer], pipe_inner_diameter: float, pipe_length: float,
                 steam_temp: float, ambient_temp: float, conv_coeff: float):
        self.layers = layers
        self.pipe_inner_diameter = pipe_inner_diameter  # mm
        self.pipe_length = pipe_length  # m
        self.steam_temp = steam_temp  # °C
        self.ambient_temp = ambient_temp  # °C
        self.conv_coeff = conv_coeff  # W/(m²·K)
        
    def calculate_performance(self) -> Dict:
        """计算方案的性能指标"""
        current_diameter = self.pipe_inner_diameter
        total_resistance = 0
        total_cost = 0
        layer_details = []
        
        for layer in self.layers:
            resistance = layer.get_thermal_resistance(current_diameter, self.pipe_length)
            cost = layer.get_cost(current_diameter, self.pipe_length)
            
            layer_details.append({
                '材料': layer.material,
                '厚度(mm)': layer.thickness,
                '内径(mm)': current_diameter,
                '外径(mm)': current_diameter + 2 * layer.thickness,
                '热阻(K/W)': resistance,
                '成本(元)': cost
            })
            
            total_resistance += resistance
            total_cost += cost
            current_diameter += 2 * layer.thickness
        
        outer_surface_area = math.pi * current_diameter * self.pipe_length / 1000  # m²
        conv_resistance = 1 / (self.conv_coeff * outer_surface_area)
        total_resistance_with_conv = total_resistance + conv_resistance
        heat_loss = (self.steam_temp - self.ambient_temp) / total_resistance_with_conv
        cost_performance = total_resistance / (total_cost / 10000)
        
        # 计算施工难度分数 (1-10分，分数越低越好)
        construction_difficulty = min(10, len(self.layers) * 2 + sum(layer.thickness for layer in self.layers) / 100)
        
        # 计算维护难度分数 (1-10分，分数越低越好)
        maintenance_difficulty = min(10, len(self.layers) * 1.5 + sum(layer.thickness for layer in self.layers) / 150)
        
        return {
            '总热阻(K/W)': total_resistance,
            '对流热阻(K/W)': conv_resistance,
            '总成本(元)': total_cost,
            '性价比((K/W)/万元)': cost_performance,
            '散热功率(W)': heat_loss,
            '施工难度': construction_difficulty,
            '维护难度': maintenance_difficulty,
            '层详情': layer_details
        }

def normalize_score(value: float, min_val: float, max_val: float, is_higher_better: bool = True) -> float:
    """归一化分数到0-1之间"""
    if min_val == max_val:
        return 1.0
    if is_higher_better:
        return (value - min_val) / (max_val - min_val)
    else:
        return (max_val - value) / (max_val - min_val)

def calculate_comprehensive_score(performance: Dict, weights: Dict, min_max_values: Dict) -> float:
    """计算方案的综合得分"""
    score = 0
    
    # 计算各指标的归一化分数
    normalized_scores = {
        '总热阻': normalize_score(
            performance['总热阻(K/W)'],
            min_max_values['总热阻(K/W)']['min'],
            min_max_values['总热阻(K/W)']['max'],
            True
        ),
        '总成本': normalize_score(
            performance['总成本(元)'],
            min_max_values['总成本(元)']['min'],
            min_max_values['总成本(元)']['max'],
            False
        ),
        '散热功率': normalize_score(
            performance['散热功率(W)'],
            min_max_values['散热功率(W)']['min'],
            min_max_values['散热功率(W)']['max'],
            False
        ),
        '施工难度': normalize_score(
            performance['施工难度'],
            min_max_values['施工难度']['min'],
            min_max_values['施工难度']['max'],
            False
        ),
        '维护难度': normalize_score(
            performance['维护难度'],
            min_max_values['维护难度']['min'],
            min_max_values['维护难度']['max'],
            False
        )
    }
    
    # 计算加权得分
    for metric, weight in weights.items():
        score += weight * normalized_scores[metric]
    
    return score

def generate_and_analyze_schemes(pipe_inner_diameter: float = 530, pipe_length: float = 2.5,
                               steam_temp: float = 271, ambient_temp: float = 25, conv_coeff: float = 2.96):
    """生成并分析保温方案"""
    best_schemes = {
        '最佳热阻': {'value': 0, 'scheme': None},
        '最低成本': {'value': float('inf'), 'scheme': None},
        '最佳性价比': {'value': 0, 'scheme': None},
        '最低散热': {'value': float('inf'), 'scheme': None},
        '综合最优': {'value': 0, 'scheme': None}
    }
    
    # 定义权重
    weights = {
        '总热阻': 0.3,    # 热阻最重要
        '总成本': 0.25,   # 成本次之
        '散热功率': 0.25, # 散热功率同样重要
        '施工难度': 0.1,  # 施工难度较次要
        '维护难度': 0.1   # 维护难度较次要
    }
    
    # 用于存储各指标的最大最小值
    min_max_values = {
        '总热阻(K/W)': {'min': float('inf'), 'max': 0},
        '总成本(元)': {'min': float('inf'), 'max': 0},
        '散热功率(W)': {'min': float('inf'), 'max': 0},
        '施工难度': {'min': float('inf'), 'max': 0},
        '维护难度': {'min': float('inf'), 'max': 0}
    }
    
    # 定义每种材料可能的厚度范围
    material_thickness_ranges = {
        '纳米气凝胶': range(10, 51, 10),      # 10-50mm
        '高温玻璃棉': range(50, 201, 50),     # 50-200mm
        '硅酸铝针刺毯': range(50, 201, 50),   # 50-200mm
        '钠镁管保护层': range(2, 7, 2),       # 2-6mm
        '钠镁硬质保温块': range(50, 151, 50)  # 50-150mm
    }
    
    all_materials = list(MATERIALS.keys())
    all_schemes = []
    total_schemes = 0
    
    # 第一遍循环：收集所有有效方案和指标范围
    for num_layers in range(2, 5):
        for materials in itertools.combinations_with_replacement(all_materials, num_layers):
            if materials[-1] not in ['钠镁管保护层', '钠镁硬质保温块']:
                continue
                
            thickness_ranges = [material_thickness_ranges[material] for material in materials]
            
            for thicknesses in itertools.product(*thickness_ranges):
                if sum(thicknesses) > 500:
                    continue
                    
                layers = [InsulationLayer(material, thickness) 
                         for material, thickness in zip(materials, thicknesses)]
                
                scheme = InsulationScheme(layers, pipe_inner_diameter, pipe_length,
                                        steam_temp, ambient_temp, conv_coeff)
                
                performance = scheme.calculate_performance()
                
                if performance['散热功率(W)'] > 500:
                    continue
                
                total_schemes += 1
                all_schemes.append((scheme, performance))
                
                # 更新指标的最大最小值
                for metric in min_max_values:
                    if metric in performance:
                        min_max_values[metric]['min'] = min(min_max_values[metric]['min'], performance[metric])
                        min_max_values[metric]['max'] = max(min_max_values[metric]['max'], performance[metric])
    
    # 第二遍循环：计算综合得分并找出最优方案
    for scheme, performance in all_schemes:
        # 更新单指标最优方案
        if performance['总热阻(K/W)'] > best_schemes['最佳热阻']['value']:
            best_schemes['最佳热阻']['value'] = performance['总热阻(K/W)']
            best_schemes['最佳热阻']['scheme'] = (scheme, performance)
        
        if performance['总成本(元)'] < best_schemes['最低成本']['value']:
            best_schemes['最低成本']['value'] = performance['总成本(元)']
            best_schemes['最低成本']['scheme'] = (scheme, performance)
        
        if performance['性价比((K/W)/万元)'] > best_schemes['最佳性价比']['value']:
            best_schemes['最佳性价比']['value'] = performance['性价比((K/W)/万元)']
            best_schemes['最佳性价比']['scheme'] = (scheme, performance)
        
        if performance['散热功率(W)'] < best_schemes['最低散热']['value']:
            best_schemes['最低散热']['value'] = performance['散热功率(W)']
            best_schemes['最低散热']['scheme'] = (scheme, performance)
        
        # 计算综合得分
        comprehensive_score = calculate_comprehensive_score(performance, weights, min_max_values)
        if comprehensive_score > best_schemes['综合最优']['value']:
            best_schemes['综合最优']['value'] = comprehensive_score
            best_schemes['综合最优']['scheme'] = (scheme, performance)
    
    print(f"\n总共生成了 {total_schemes} 个有效方案")
    print("\n" + "="*50)
    
    # 打印最优方案
    for criterion, data in best_schemes.items():
        scheme, performance = data['scheme']
        print(f"\n{criterion}方案:")
        print("-" * 30)
        
        # 打印层结构
        print("层结构:")
        total_thickness = 0
        for layer in performance['层详情']:
            print(f"  - {layer['材料']}: {layer['厚度(mm)']}mm")
            total_thickness += layer['厚度(mm)']
        print(f"总厚度: {total_thickness}mm")
        
        # 打印性能指标
        print(f"总热阻: {performance['总热阻(K/W)']:.2f} K/W")
        print(f"总成本: {performance['总成本(元)']:.2f} 元")
        print(f"性价比: {performance['性价比((K/W)/万元)']:.2f} (K/W)/万元")
        print(f"散热功率: {performance['散热功率(W)']:.2f} W")
        print(f"施工难度: {performance['施工难度']:.1f}/10")
        print(f"维护难度: {performance['维护难度']:.1f}/10")
        
        # 如果是综合最优方案，显示综合得分
        if criterion == '综合最优':
            print(f"综合得分: {data['value']:.3f}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    generate_and_analyze_schemes() 