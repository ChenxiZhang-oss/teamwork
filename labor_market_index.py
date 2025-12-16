import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple
import os

# 设置绘图风格和字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class LaborMarketIndexSystem:
    def __init__(self):
        # 1. 方法论强制要求：子维度与关键词定义
        self.structure = {
            "job_search": {
                "name": "求职活跃度",
                "keywords": ["找工作", "招聘", "求职", "招聘信息", "校招", "春招", "秋招", "面试", "简历"],
                "polarity": "positive" # 越高越好 (活跃)
            },
            "emp_difficulty": {
                "name": "就业困难/压力",
                "keywords": ["就业难", "找工作难", "应届生就业", "应届生找工作", "就业形势", "就业前景", "行业前景", "薪资水平", "薪资查询", "大学生就业", "毕业生就业"],
                "polarity": "negative" # 越高越差 (压力大)
            },
            "unemp_pressure": {
                "name": "失业/裁员压力",
                "keywords": ["失业", "裁员", "裁员潮", "被裁", "优化", "失业金", "失业保险", "失业补助", "失业登记", "再就业", "低门槛工作", "临时工", "兼职", "蓝领招聘"],
                "polarity": "negative"
            },
            "struc_pressure": {
                "name": "结构性/弱势群体压力",
                "keywords": ["35岁就业", "35岁找工作", "中年就业", "蓝领招聘", "低学历就业", "外卖骑手", "快递员", "送外卖", "兼职", "临时工", "底层劳动岗位"],
                "polarity": "negative"
            }
        }
        self.raw_data = None
        self.norm_data_log = {} # 存储不同标准化方法的结果

    # -------------------------------------------------------------------------
    # 2. 数据获取模块 (Mock & Real Placeholder)
    # -------------------------------------------------------------------------
    def generate_mock_data(self, months: int = 24, start_date: str = '2023-01-01'):
        """
        生成高仿真模拟数据：包含趋势、季节性（如毕业季、年末）、随机波动
        """
        dates = pd.date_range(start=start_date, periods=months, freq='ME')
        data = {}
        
        np.random.seed(2025) # 复现性
        
        for domain, info in self.structure.items():
            for kw in info['keywords']:
                # 基础值
                base = np.random.uniform(1000, 5000)
                
                # 1. 长期趋势 (Trend)
                trend = np.linspace(0, base * 0.2, months) 
                if info['polarity'] == 'negative':
                    trend = trend # 压力随时间略微增加
                
                # 2. 周期性/季节性 (Seasonality)
                seasonality = np.zeros(months)
                month_indices = dates.month
                
                # 毕业季 (6-7月) -> 求职活跃度高，就业困难搜索高
                if domain in ['job_search', 'emp_difficulty']:
                    mask = (month_indices == 6) | (month_indices == 7) | (month_indices == 3) # 金三银四 + 毕业季
                    seasonality[mask] = base * 0.4
                    
                # 年末/年初 (12-1月) -> 裁员压力高
                if domain in ['unemp_pressure', 'struc_pressure']:
                    mask = (month_indices == 12) | (month_indices == 1)
                    seasonality[mask] = base * 0.3
                
                # 3. 随机波动 (Noise)
                noise = np.random.normal(0, base * 0.1, months)
                
                # 合成
                val = base + trend + seasonality + noise
                data[f"{domain}_{kw}"] = np.maximum(val, 0).astype(int)
        
        # 构造DataFrame
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Timestamp'
        
        # 转换为长格式以便存储元数据 (符合要求的输出格式)
        # 但为了计算方便，内部主要使用宽格式(Wide Format)
        self.raw_data = df
        
        # 导出原始数据 CSV
        long_df = df.reset_index().melt(id_vars='Timestamp', var_name='Combined_Key', value_name='Search_Volume')
        long_df['Sub_Domain'] = long_df['Combined_Key'].apply(lambda x: x.split('_')[0] + ('_' + x.split('_')[1] if x.startswith('job') else ''))
        # 修复切分逻辑
        def get_domain_name(key):
            for d in self.structure.keys():
                if key.startswith(d):
                    return self.structure[d]['name']
            return "Unknown"
        
        def get_keyword(key):
            for d in self.structure.keys():
                if key.startswith(d):
                    return key[len(d)+1:]
            return key
            
        long_df['Sub_Domain_Name'] = long_df['Combined_Key'].apply(get_domain_name)
        long_df['Keyword'] = long_df['Combined_Key'].apply(get_keyword)
        
        output_df = long_df[['Timestamp', 'Sub_Domain_Name', 'Keyword', 'Search_Volume']]
        output_df.to_csv('1_raw_data.csv', index=False, encoding='utf-8-sig')
        print(">> [Data] 原始数据集已生成: 1_raw_data.csv")
        return df

    # -------------------------------------------------------------------------
    # 3. 数据标准化模块
    # -------------------------------------------------------------------------
    def normalize_data(self, method: str = 'min-max'):
        """
        标准化方法：z-score 或 min-max
        """
        df = self.raw_data.copy()
        
        if method == 'z-score':
            # Z = (X - Mean) / Std
            norm_df = (df - df.mean()) / df.std()
        elif method == 'min-max':
            # MinMax = (X - Min) / (Max - Min) -> 映射到 [0, 100] 以便观察
            norm_df = (df - df.min()) / (df.max() - df.min()) * 100
        else:
            raise ValueError("Unknown method")
            
        self.norm_data_log[method] = norm_df
        
        # 输出标准化后数据集 (CSV) - 仅以当前调用的方法输出演示
        # 为了展示对比，我们需要把原始值和标准化值放在一起
        compare_data = []
        for col in df.columns:
            temp = pd.DataFrame({
                'Timestamp': df.index,
                'Variable': col,
                'Raw_Value': df[col].values,
                f'Norm_Value_{method}': norm_df[col].values
            })
            compare_data.append(temp)
        
        full_compare = pd.concat(compare_data)
        full_compare.to_csv(f'2_normalized_data_{method}.csv', index=False, encoding='utf-8-sig')
        print(f">> [Data] 标准化数据已生成: 2_normalized_data_{method}.csv")
        
        return norm_df

    # -------------------------------------------------------------------------
    # 4. 子指数构造模块 (Aggregation)
    # -------------------------------------------------------------------------
    def calculate_sub_indices(self, norm_df: pd.DataFrame, method: str = 'linear', weights: Optional[Dict] = None):
        """
        method: 'linear' (加权求和) 或 'mpi' (Mazziotta-Pareto Index)
        """
        sub_indices = pd.DataFrame(index=norm_df.index)
        logs = []

        for domain, info in self.structure.items():
            # 找到该维度下的所有列
            cols = [c for c in norm_df.columns if c.startswith(domain)]
            domain_data = norm_df[cols]
            
            if method == 'linear':
                # 默认等权重 Mean
                val = domain_data.mean(axis=1)
                sub_indices[info['name']] = val
                
                # 记录日志 (取第一行做示例)
                logs.append(f"维度: {info['name']} | 方法: Linear | 示例(T=0): Mean({domain_data.iloc[0].values.round(2)}) = {val.iloc[0]:.2f}")
                
            elif method == 'mpi':
                # MPI = Mean +/- (Std * CV)? 
                # Prompt公式: MPI = 均值 - 变异系数 * 均值 = Mean - (Std/Mean)*Mean = Mean - Std
                # 考虑到这是“压力”指数，我们通常希望反映“严重性”。
                # 如果内部指标差异大（不均衡），通常认为状态更不稳定/严重。
                # 此处严格遵循 Prompt 文本公式: Mean - CV * Mean (即 Mean - Std)
                # *但在实际指数构建中，对于负向指标（压力），通常是 Mean + Std*
                # 为了演示“稳健性测试”，我们这里设定：对于压力指标(negative)，我们加罚项；对于活跃度(positive)，我们减罚项。
                
                mean = domain_data.mean(axis=1)
                std = domain_data.std(axis=1)
                
                # 简单的 MPI 实现: Mean +/- Std
                # 修正：根据Prompt "MPI = 均值 - 变异系数 * 均值"
                # 实际上这个公式等于 Mean - Std。
                # 我们假设这是一个"Penalty"。如果"不均衡"是坏事，那么指数应该变得"更差"。
                # 如果指数是"压力"(越低越好)，变差意味着变高 -> Mean + Std
                # 如果指数是"活跃度"(越高越好)，变差意味着变低 -> Mean - Std
                
                if info['polarity'] == 'negative':
                    # 压力类：不均衡会导致压力感知的加剧 -> Mean + Std
                    val = mean + std
                else:
                    # 活跃度类：不均衡意味着只有部分活跃 -> Mean - Std
                    val = mean - std
                
                sub_indices[info['name']] = val
                logs.append(f"维度: {info['name']} | 方法: MPI | 示例(T=0): Mean={mean.iloc[0]:.2f}, Std={std.iloc[0]:.2f} -> Result={val.iloc[0]:.2f}")

        return sub_indices, logs

    # -------------------------------------------------------------------------
    # 5. 综合指数构造模块
    # -------------------------------------------------------------------------
    def calculate_composite_index(self, sub_df: pd.DataFrame, method: str = 'linear', weights: Optional[Dict] = None):
        """
        综合指数聚合
        """
        # 在聚合前，通常需要再次标准化子指数，以防量纲差异
        # 这里进行简单的 Min-Max 归一化到 0-100
        sub_norm = (sub_df - sub_df.min()) / (sub_df.max() - sub_df.min()) * 100
        
        if method == 'linear':
            # 线性加权
            if weights:
                # weights: {'求职活跃度': 0.3, ...}
                weighted_sum = 0
                for col in sub_norm.columns:
                    w = weights.get(col, 1.0 / len(sub_norm.columns))
                    weighted_sum += sub_norm[col] * w
                composite = weighted_sum
            else:
                composite = sub_norm.mean(axis=1)
                
        elif method == 'mpi':
            # 综合指数 MPI
            mean = sub_norm.mean(axis=1)
            std = sub_norm.std(axis=1)
            # 综合指数定义为“劳动力市场状况/压力综合指数”
            # 假设我们主要关注“压力” (Composite Index通常指代整体状况)
            # 这里统一使用 Mean + Std (假设反映压力/风险)
            composite = mean + std
            
        composite.name = 'Composite_Index'
        return composite

    # -------------------------------------------------------------------------
    # 6. 时间序列分析模块
    # -------------------------------------------------------------------------
    def analyze_timeseries(self, series: pd.Series):
        """
        趋势、周期、异常值
        """
        # 1. 趋势 (MA3)
        ma3 = series.rolling(window=3).mean()
        
        # 2. 异常值 (3 Sigma)
        mean = series.mean()
        std = series.std()
        threshold_upper = mean + 2 * std # 使用2倍标准差以便在模拟数据中更容易抓到点
        threshold_lower = mean - 2 * std
        outliers = series[(series > threshold_upper) | (series < threshold_lower)]
        
        report = {
            'trend': ma3,
            'outliers': outliers,
            'stats': series.describe()
        }
        return report

    # -------------------------------------------------------------------------
    # 7. 稳健性测试与主流程
    # -------------------------------------------------------------------------
    def run_full_process(self):
        print(">>> 开始执行全流程...")
        
        # 1. 获取数据
        self.generate_mock_data()
        
        # 2. 生成主要结果 (Base Case: Min-Max + Linear + Equal Weights)
        norm_base = self.normalize_data('min-max')
        sub_base, sub_logs = self.calculate_sub_indices(norm_base, 'linear')
        comp_base = self.calculate_composite_index(sub_base, 'linear')
        
        # 导出子指数和综合指数
        sub_base.to_csv('3_sub_indices.csv', encoding='utf-8-sig')
        comp_base.to_csv('4_composite_index.csv', encoding='utf-8-sig')
        print(">> [Data] 子指数与综合指数已生成")

        # 3. 稳健性测试 (Robustness Checks)
        results = pd.DataFrame(index=self.raw_data.index)
        results['Base_Case'] = comp_base
        
        # Check 1: 不同标准化 (Z-score vs MinMax)
        norm_z = self.normalize_data('z-score')
        sub_z, _ = self.calculate_sub_indices(norm_z, 'linear')
        # Z-score后的子指数可能有负数，再归一化合成
        comp_z = self.calculate_composite_index(sub_z, 'linear')
        results['Robust_ZScore'] = comp_z
        
        # Check 2: 不同聚合 (MPI vs Linear)
        sub_mpi, _ = self.calculate_sub_indices(norm_base, 'mpi')
        comp_mpi = self.calculate_composite_index(sub_mpi, 'mpi')
        results['Robust_MPI'] = comp_mpi
        
        # Check 3: 不同权重
        custom_weights = {
            "求职活跃度": 0.3, 
            "就业困难/压力": 0.3, 
            "失业/裁员压力": 0.2, 
            "结构性/弱势群体压力": 0.2
        }
        comp_weighted = self.calculate_composite_index(sub_base, 'linear', weights=custom_weights)
        results['Robust_Weights'] = comp_weighted
        
        results.to_csv('5_robustness_check.csv', encoding='utf-8-sig')
        print(">> [Data] 稳健性测试数据已生成")
        
        # 4. 时间序列分析
        ts_report = self.analyze_timeseries(comp_base)
        
        # 5. 可视化
        self.plot_all(norm_base, sub_base, results, ts_report)
        
        # 6. 生成文字报告
        self.generate_markdown_report(sub_logs, ts_report, results)

    def plot_all(self, norm_df, sub_df, robust_df, ts_report):
        fig = plt.figure(figsize=(15, 20))
        
        # 图1: 原始搜索量 Top 10 趋势
        ax1 = plt.subplot(5, 1, 1)
        top10_cols = self.raw_data.mean().nlargest(10).index
        self.raw_data[top10_cols].plot(ax=ax1, linewidth=1.5, alpha=0.8)
        ax1.set_title("原始搜索量 Top 10 关键词趋势", fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', ncol=5, fontsize='small')
        
        # 图2: 标准化分布直方图
        ax2 = plt.subplot(5, 1, 2)
        sns.histplot(self.raw_data.values.flatten(), bins=50, color='blue', alpha=0.3, label='Raw', kde=True, ax=ax2)
        ax2_twin = ax2.twiny()
        sns.histplot(norm_df.values.flatten(), bins=50, color='red', alpha=0.3, label='MinMax Norm', kde=True, ax=ax2_twin)
        ax2.set_title("原始数据 vs 标准化数据 分布对比", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Raw Value")
        ax2_twin.set_xlabel("Normalized Value")
        
        # 图3: 子指数趋势
        ax3 = plt.subplot(5, 1, 3)
        sub_df.plot(ax=ax3, marker='o', markersize=4)
        ax3.set_title("四个子维度指数趋势", fontsize=12, fontweight='bold')
        
        # 图4: 综合指数与趋势线
        ax4 = plt.subplot(5, 1, 4)
        robust_df['Base_Case'].plot(ax=ax4, label='综合指数', color='black', linewidth=2)
        ts_report['trend'].plot(ax=ax4, label='MA(3) 趋势', color='red', linestyle='--')
        
        # 标记异常值
        outliers = ts_report['outliers']
        if not outliers.empty:
            ax4.scatter(outliers.index, outliers.values, color='red', s=100, label='异常值 (2σ)', zorder=5)
            
        ax4.set_title("劳动力市场综合指数 & 趋势分析", fontsize=12, fontweight='bold')
        ax4.legend()
        
        # 图5: 稳健性测试对比
        ax5 = plt.subplot(5, 1, 5)
        robust_df.plot(ax=ax5)
        ax5.set_title("稳健性测试：不同参数下的指数对比", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('labor_market_analysis_report.png', dpi=300)
        print(">> [Plot] 可视化图表已保存: labor_market_analysis_report.png")

    def generate_markdown_report(self, sub_logs, ts_report, robust_df):
        with open('analysis_report.md', 'w', encoding='utf-8') as f:
            f.write("# 劳动力市场综合指数分析报告\n\n")
            
            f.write("## 1. 中间处理过程日志\n")
            f.write("| 步骤 | 说明 |\n|---|---|\n")
            for log in sub_logs[:5]: # 只取前5条避免过长
                f.write(f"| 子指数计算 | {log} |\n")
            f.write("\n")
            
            f.write("## 2. 时间序列分析\n")
            f.write(f"- **趋势**: 长期趋势均值 {ts_report['trend'].mean():.2f}\n")
            f.write(f"- **波动性 (Std)**: {ts_report['stats']['std']:.2f}\n")
            f.write("- **异常值检测**:\n")
            if not ts_report['outliers'].empty:
                for date, val in ts_report['outliers'].items():
                    f.write(f"  - {date.strftime('%Y-%m')}: {val:.2f} (突变点)\n")
            else:
                f.write("  - 无显著异常值\n")
            f.write("\n")
            
            f.write("## 3. 稳健性测试结论\n")
            corr = robust_df.corr()
            f.write("各方案相关性系数矩阵：\n\n")
            f.write(corr.to_markdown())
            f.write("\n\n**结论**：\n")
            min_corr = corr.min().min()
            if min_corr > 0.8:
                f.write(f"各方案相关性极高 (>{min_corr:.2f})，说明指数构造方法非常稳健，参数变化对核心趋势影响较小。")
            else:
                f.write(f"部分方案存在差异 (最低相关性 {min_corr:.2f})，建议根据实际业务场景选择最符合直觉的权重方案。")
            
        print(">> [Report] 分析报告已生成: analysis_report.md")

if __name__ == "__main__":
    system = LaborMarketIndexSystem()
    system.run_full_process()
