import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from main import misoa

# ==========================================
# CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="MISOA Routing Optimization", page_icon="🚚", layout="wide")
st.title("🚚 Ứng dụng MISOA: Tối ưu hóa đường đi (Bài toán TSP)")

st.markdown("""
<style>
    /* Độ lại các thẻ Metric (Số liệu) */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E; /* Màu nền tối sang trọng */
        border: 1px solid #333344;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px); /* Hiệu ứng nảy lên khi di chuột */
        border-color: #00FFCC; /* Sáng viền màu Neon */
    }
    /* Chỉnh màu chữ của Metric */
    div[data-testid="metric-container"] label {
        color: #A6ACCD !important;
        font-weight: bold;
    }
    div[data-testid="metric-container"] div {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("Sử dụng Đàn mòng biển cải tiến để tìm đường đi ngắn nhất qua tất cả các điểm giao hàng và quay về gốc.")

# ==========================================
# HÀM HỖ TRỢ BÀI TOÁN TSP
# ==========================================
def calc_distance(route, cities):
    """Tính tổng chiều dài quãng đường"""
    dist = 0.0
    for i in range(len(route) - 1):
        dist += np.linalg.norm(cities[route[i]] - cities[route[i+1]])
    dist += np.linalg.norm(cities[route[-1]] - cities[route[0]]) 
    return dist

def plot_route(cities, route, title, color='#00FFCC'):
    """Vẽ bản đồ tuyến đường (Phong cách Radar GPS Nền Tối)"""
    plt.style.use('dark_background') 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.grid(True, color='#333333', linestyle='--', alpha=0.7)
    
    ax.scatter(cities[:, 0], cities[:, 1], c='#FF003C', s=60, edgecolors='white', zorder=5)
    
    ax.scatter(cities[route[0], 0], cities[route[0], 1], c='#00FFCC', s=250, marker='*', edgecolors='white', zorder=6, label='Trạm trung chuyển')
    
    ordered_cities = cities[route]
    ordered_cities = np.vstack((ordered_cities, cities[route[0]])) 
    ax.plot(ordered_cities[:, 0], ordered_cities[:, 1], c=color, linestyle='-', linewidth=2.5, alpha=0.8)
    
    for i, txt in enumerate(range(len(cities))):
        ax.annotate(txt, (cities[i, 0] + 1.5, cities[i, 1] + 1.5), color='white', fontsize=10, fontweight='bold')
        
    ax.set_title(title, color='white', fontweight='bold', fontsize=14)

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333333')
        
    ax.legend(facecolor='#0E1117', edgecolor='#333333', labelcolor='white')
    return fig

# ==========================================
# GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.sidebar.header("⚙️ Cấu hình hệ thống")
num_cities = st.sidebar.slider("Số lượng điểm giao hàng (Cities)", 10, 50, 20)
search_agents = st.sidebar.slider("Số lượng mòng biển (Agents)", 10, 100, 30)
max_iterations = st.sidebar.slider("Số vòng lặp (Max Iterations)", 50, 500, 200)
randomize_btn = st.sidebar.button("🎲 Đổi bản đồ ngẫu nhiên")

if 'cities' not in st.session_state or len(st.session_state.cities) != num_cities or randomize_btn:
    st.session_state.cities = np.random.rand(num_cities, 2) * 100
    st.toast("Đã xáo trộn bản đồ thành công! 🎲", icon="✅")

cities = st.session_state.cities
naive_route = np.arange(num_cities)
naive_score = calc_distance(naive_route, cities)

if st.sidebar.button("🚀 Chạy Tối Ưu Hóa", type="primary"):
    st.markdown("### 🏃‍♂️ Đang điều phối bầy mòng biển tìm đường...")
    progress_bar = st.progress(0)
    
    # ---------------------------------------------------------
    # HÀM MỤC TIÊU (FITNESS) DÀNH CHO BÀI TOÁN TSP
    # ---------------------------------------------------------
    def objective_function(position):
        route = np.argsort(position)
        return calc_distance(route, cities)

    # ---------------------------------------------------------
    # CHẠY MISOA
    # ---------------------------------------------------------
    dimension = num_cities
    lower_bound = -10.0
    upper_bound = 10.0
    
    best_score, best_pos, convergence_curve = misoa(
        search_agents, max_iterations, lower_bound, upper_bound, dimension, objective_function
    )
    progress_bar.progress(100)
    
    best_route = np.argsort(best_pos)
    
    naive_route = np.arange(num_cities)
    naive_score = calc_distance(naive_route, cities)
    
    # ---------------------------------------------------------
    # HIỂN THỊ KẾT QUẢ VÀ TÍNH TIỀN XĂNG
    # ---------------------------------------------------------
    st.success("✅ Thuật toán đã hội tụ thành công!")
    
    dist_saved = naive_score - best_score
    fuel_per_100km = 8.0 # Xe tải nhỏ tốn 8 lít/100km
    gas_price = 24000    # Giá xăng 24,000 VNĐ/lít
    
    fuel_saved = dist_saved * (fuel_per_100km / 100) # Số lít xăng tiết kiệm
    money_saved = fuel_saved * gas_price             # Số tiền tiết kiệm
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("❌ Lộ trình ngẫu nhiên")
        st.metric("Tổng chiều dài", f"{naive_score:.2f} km")
        fig_naive = plot_route(cities, naive_route, "Lộ trình chưa tối ưu", color='gray')
        st.pyplot(fig_naive)
        
    with col2:
        st.success("✨ Lộ trình tối ưu (MISOA)")
        st.metric("Tổng chiều dài", f"{best_score:.2f} km", f"-{dist_saved:.2f} km (Tiết kiệm)", delta_color="inverse")
        fig_misoa = plot_route(cities, best_route, "Lộ trình MISOA tối ưu", color='blue')
        st.pyplot(fig_misoa)
        
    with col3:
        st.warning("💰 Hiệu quả kinh tế (1 Chuyến)")
        st.metric("⛽ Xăng tiết kiệm", f"{fuel_saved:.1f} Lít", "Bảo vệ môi trường")
        st.metric("💵 Tiền tiết kiệm", f"{money_saved:,.0f} VNĐ", "Tăng lợi nhuận")
        
        st.markdown(f"> *Nếu chạy 30 chuyến/tháng, MISOA giúp công ty tiết kiệm **{(money_saved * 30):,.0f} VNĐ** chi phí vận hành!*")
    
    st.write("### 🧬 Thứ tự các trạm giao hàng cần đi:")
    route_str = " ➔ ".join([str(x) for x in best_route]) + f" ➔ {best_route[0]}"
    st.code(route_str)
else:
    st.info("👈 Hãy bấm 'Chạy Tối Ưu Hóa' bên thanh công cụ để tìm đường đi ngắn nhất!")
    st.markdown("### 🗺️ Bản đồ các trạm giao hàng hiện tại (Chưa tối ưu)")
    
    fig_raw = plot_route(cities, naive_route, "Lộ trình ngẫu nhiên sinh ra", color='gray')
    st.pyplot(fig_raw)