import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# ==========================================
def F1(x):
    """Hàm mục tiêu F1 (Sphere Function)"""
    return np.sum(x**2)

def fun_info(F):
    if F == 'F1':
        lowerbound = -100
        upperbound = 100
        dimension = 30
        fitness = F1
        return lowerbound, upperbound, dimension, fitness


# ==========================================
# ==========================================
def init(search_agents, dimension, upperbound, lowerbound):
    """Khởi tạo vị trí ngẫu nhiên cho bầy mòng biển"""
    if isinstance(upperbound, (int, float)):
        pos = np.random.uniform(lowerbound, upperbound, (search_agents, dimension))
    else:
        pos = np.zeros((search_agents, dimension))
        for i in range(dimension):
            pos[:, i] = np.random.uniform(lowerbound[i], upperbound[i], search_agents)
    return pos

# ==========================================
# HÀM KHỞI TẠO CẢI TIẾN (CHAOTIC MAP)
# ==========================================
def chaotic_init(search_agents, dimension, upper_bound, lower_bound):
    """Khởi tạo bầy mòng biển bằng Chaos Logistic Map"""
    x = np.random.rand(search_agents, dimension)
    x[x == 0] = 0.001 
    
    a = 4.0
    for _ in range(50):
        x = a * x * (1 - x)
        
    positions = np.zeros((search_agents, dimension))
    if isinstance(upper_bound, (int, float)):
        positions = lower_bound + x * (upper_bound - lower_bound)
    else:
        for i in range(dimension):
            positions[:, i] = lower_bound[i] + x[:, i] * (upper_bound[i] - lower_bound[i])
            
    return positions

# ==========================================
# ==========================================
def soa(search_agents, max_iterations, lower_bound, upper_bound, dimension, objective):
    position = np.zeros(dimension)
    score = float('inf') 
    
    # Khởi tạo quần thể
    positions = init(search_agents, dimension, upper_bound, lower_bound)
    convergence = np.zeros(max_iterations)
    
    l = 0
    while l < max_iterations:
        for i in range(search_agents):  
            # Kiểm tra và giới hạn không gian tìm kiếm (Boundary check)
            positions[i, :] = np.clip(positions[i, :], lower_bound, upper_bound)
            
            # Tính toán độ thích nghi (Fitness)
            fitness = objective(positions[i, :])
            
            # Cập nhật mòng biển tốt nhất
            if fitness < score: 
                score = fitness 
                position = positions[i, :].copy()
                
        # Tính toán biến Fc (giảm dần từ 2 về 0)
        Fc = 2 - l * (2 / max_iterations) 
        
        for i in range(search_agents):
            for j in range(dimension):     
                r1 = np.random.rand() 
                r2 = np.random.rand() 
                
                # Quá trình Di cư (Khám phá)
                A1 = 2 * Fc * r1 - Fc 
                C1 = 2 * r2 
                b = 1             
                
                # Quá trình Tấn công (Khai thác theo hình xoắn ốc)
                ll = (Fc - 1) * np.random.rand() + 1  
                D_alphs = Fc * positions[i, j] + A1 * (position[j] - positions[i, j])                   
                X1 = D_alphs * np.exp(b * ll) * np.cos(ll * 2 * np.pi) + position[j]
                
                # Cập nhật vị trí mới
                positions[i, j] = X1
                
        convergence[l] = score
        l += 1    
        
    return score, position, convergence

# ==========================================
# THUẬT TOÁN CẢI TIẾN (MISOA - BƯỚC 1)
# ==========================================
def misoa(search_agents, max_iterations, lower_bound, upper_bound, dimension, objective):
    position = np.zeros(dimension)
    score = float('inf') 
    
    positions = chaotic_init(search_agents, dimension, upper_bound, lower_bound)
    convergence = np.zeros(max_iterations)
    
    l = 0
    while l < max_iterations:
        for i in range(search_agents):  
            positions[i, :] = np.clip(positions[i, :], lower_bound, upper_bound)
            fitness = objective(positions[i, :])
            
            if fitness < score: 
                score = fitness 
                position = positions[i, :].copy()
                
        Fc = 2 * (1 - np.sin((np.pi / 2) * (l / max_iterations)))
        
        for i in range(search_agents):
            for j in range(dimension):     
                r1 = np.random.rand() 
                r2 = np.random.rand() 
                
                A1 = 2 * Fc * r1 - Fc 
                C1 = 2 * r2 
                b = 1             
                
                ll = (Fc - 1) * np.random.rand() + 1  
                D_alphs = Fc * positions[i, j] + A1 * (position[j] - positions[i, j])                   
                X1 = D_alphs * np.exp(b * ll) * np.cos(ll * 2 * np.pi) + position[j]
                
                positions[i, j] = X1
                
        convergence[l] = score
        l += 1    
        
    return score, position, convergence

# ==========================================
# ==========================================
if __name__ == "__main__":
    search_agents = 30 
    fun_name = 'F1'  
    max_iterations = 1000 
    
    lowerbound, upperbound, dimension, fitness = fun_info(fun_name)
    
    # 1. Chạy SOA gốc
    print(f"Đang chạy SOA gốc...")
    best_score_soa, _, soa_curve = soa(search_agents, max_iterations, lowerbound, upperbound, dimension, fitness)
    
    # 2. Chạy MISOA cải tiến
    print(f"Đang chạy MISOA cải tiến (Chaotic Init)...")
    best_score_misoa, _, misoa_curve = misoa(search_agents, max_iterations, lowerbound, upperbound, dimension, fitness)
    
    print(f"\nGiá trị tối ưu tốt nhất của SOA:   {best_score_soa}")
    print(f"Giá trị tối ưu tốt nhất của MISOA: {best_score_misoa}")

    # Vẽ biểu đồ kết quả
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x = np.linspace(lowerbound, upperbound, 50)
    y = np.linspace(lowerbound, upperbound, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 
    ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title('Objective space (F1 Function)')
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.set_zlabel('F1(x_1, x_2)')

    # Vẽ cả 2 đường cong trên cùng 1 biểu đồ
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(soa_curve, color='red', linewidth=2, label='SOA (Gốc)') 
    ax2.plot(misoa_curve, color='blue', linewidth=2, linestyle='--', label='MISOA (Cải tiến)') 
    ax2.set_title('Convergence Curve Comparison')
    ax2.set_xlabel('Iterations (Vòng lặp)')
    ax2.set_ylabel('Best score (Giá trị tốt nhất)')
    ax2.set_yscale('log') # Dùng thang đo logarit để thấy rõ sự khác biệt ở các số thập phân siêu nhỏ
    ax2.grid(True, which="both", ls="--")
    ax2.legend()

    plt.tight_layout()
    plt.show()