import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from main import soa, misoa, fun_info

def run_benchmark():
    runs = 30
    search_agents = 30
    max_iterations = 500
    functions = ['F1', 'F2', 'F9', 'F10']

    all_results = []

    for func in functions:
        print(f"Đang kiểm thử hàm {func} ({runs} lần chạy độc lập)...")
        lowerbound, upperbound, dimension, fitness = fun_info(func)

        soa_scores = []
        misoa_scores = []

        for i in range(runs):
            #SOA
            score_soa, _, _ = soa(search_agents, max_iterations, lowerbound, upperbound, dimension, fitness)
            soa_scores.append(score_soa)
            
            #MISOA
            score_misoa, _, _ = misoa(search_agents, max_iterations, lowerbound, upperbound, dimension, fitness)
            misoa_scores.append(score_misoa)

        
        all_results.append({
            "Hàm": func,
            "Thuật toán": "SOA",
            "Best": np.min(soa_scores),
            "Worst": np.max(soa_scores),
            "Mean (Trung bình)": np.mean(soa_scores),
            "Std (Độ lệch chuẩn)": np.std(soa_scores)
        })

        
        all_results.append({
            "Hàm": func,
            "Thuật toán": "MISOA",
            "Best": np.min(misoa_scores),
            "Worst": np.max(misoa_scores),
            "Mean (Trung bình)": np.mean(misoa_scores),
            "Std (Độ lệch chuẩn)": np.std(misoa_scores)
        })

    df = pd.DataFrame(all_results)
    
    print("\nKẾT QUẢ BENCHMARK THỐNG KÊ:")
    print(df.to_string(index=False))

    output_folder = os.path.join(parent_dir, "results", "excel_reports")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "Benchmark_Results.xlsx")
    
    df.to_excel(output_file, index=False)
    print(f"\nĐã xuất kết quả thành công ra file:\n ->{output_file}")

if __name__ == "__main__":
    run_benchmark()