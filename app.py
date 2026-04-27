import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import misoa, soa

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

def plot_route(cities, route, title, color='#4682B4'):
    """Vẽ bản đồ tuyến đường"""
    plt.style.use('default') 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA') 
    
    ax.grid(True, color='#E0E0E0', linestyle='--', alpha=0.7)
    
    ax.scatter(cities[:, 0], cities[:, 1], c='#FF7F50', s=50, edgecolors='white', zorder=5)
    
    ax.scatter(cities[route[0], 0], cities[route[0], 1], c='#FFD700', s=200, marker='*', edgecolors='#FF8C00', zorder=6, label='Trạm trung chuyển')
    
    ordered_cities = cities[route]
    ordered_cities = np.vstack((ordered_cities, cities[route[0]])) 
    ax.plot(ordered_cities[:, 0], ordered_cities[:, 1], c=color, linestyle='-', linewidth=2.5, alpha=0.85)
    
    for i, txt in enumerate(range(len(cities))):
        ax.annotate(txt, (cities[i, 0] + 1.2, cities[i, 1] + 1.2), color='#444444', fontsize=9, fontweight='bold')
        
    ax.set_title(title, color='#333333', fontweight='bold', fontsize=14)
    
    ax.tick_params(colors='#555555')
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
        
    ax.legend(facecolor='white', edgecolor='#CCCCCC', labelcolor='#333333')
    return fig

# ==========================================
# GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.sidebar.header("⚙️ Cấu hình hệ thống")
num_cities = st.sidebar.slider("Số lượng điểm giao hàng (Cities)", 10, 50, 20)
search_agents = st.sidebar.slider("Số lượng mòng biển (Agents)", 10, 100, 30)
max_iterations = st.sidebar.slider("Số vòng lặp (Max Iterations)", 50, 500, 200)

st.sidebar.markdown("---")
st.sidebar.header("📊 Chế độ chạy")
run_mode = st.sidebar.radio("Lựa chọn phương thức:", ["Chạy 1 lần (Vẽ bản đồ trực quan)", "Chạy nhiều lần (Đánh giá thống kê)"])

if run_mode == "Chạy nhiều lần (Đánh giá thống kê)":
    num_runs = st.sidebar.number_input("Số lần chạy độc lập (Runs)", min_value=5, max_value=50, value=10)

st.sidebar.markdown("---")
randomize_btn = st.sidebar.button("🎲 Đổi bản đồ ngẫu nhiên")

if 'cities' not in st.session_state or len(st.session_state.cities) != num_cities or randomize_btn:
    st.session_state.cities = np.random.rand(num_cities, 2) * 100
    st.toast("Đã xáo trộn bản đồ thành công! 🎲", icon="✅")

cities = st.session_state.cities
naive_route = np.arange(num_cities)
naive_score = calc_distance(naive_route, cities)

# ==========================================
# XỬ LÝ CHẠY THUẬT TOÁN (SOA VÀ MISOA)
# ==========================================
if st.sidebar.button("🚀 Bắt đầu Tối Ưu Hóa", type="primary"):
    def objective_function(position):
        route = np.argsort(position)
        return calc_distance(route, cities)

    dimension = num_cities
    lower_bound = -10.0
    upper_bound = 10.0
    
    # ==========================================
    # CHẾ ĐỘ 1: CHẠY 1 LẦN (TRỰC QUAN)
    # ==========================================
    if run_mode == "Chạy 1 lần (Vẽ bản đồ trực quan)":
        # 1. Chạy SOA nguyên bản
        st.markdown("### 🐢 Đang chạy SOA (Bản gốc)...")
        progress_bar = st.progress(0)
        score_soa, pos_soa, curve_soa = soa(
            search_agents, max_iterations, lower_bound, upper_bound, dimension, objective_function
        )
        progress_bar.progress(50)
        
        # 2. Chạy MISOA cải tiến
        st.markdown("### 🦅 Đang chạy MISOA (Cải tiến)...")
        best_score, best_pos, curve_misoa = misoa(
            search_agents, max_iterations, lower_bound, upper_bound, dimension, objective_function
        )
        progress_bar.progress(100)
        
        best_route = np.argsort(best_pos)
        
        # --- HIỂN THỊ KẾT QUẢ VÀ TÍNH TIỀN XĂNG ---
        st.success("✅ Cả 2 Thuật toán đã hội tụ thành công!")
        
        dist_saved = naive_score - best_score
        fuel_per_100km = 8.0 
        gas_price = 24000    
        
        fuel_saved = dist_saved * (fuel_per_100km / 100) 
        money_saved = fuel_saved * gas_price             
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("❌ Lộ trình chưa tối ưu")
            st.metric("Tổng chiều dài", f"{naive_score:.2f} km")
            fig_naive = plot_route(cities, naive_route, "Lộ trình chưa tối ưu", color='gray')
            st.pyplot(fig_naive)
            
        with col2:
            st.success("✨ Lộ trình tối ưu (MISOA)")
            st.metric("Tổng chiều dài", f"{best_score:.2f} km", f"-{dist_saved:.2f} km (Tiết kiệm)", delta_color="inverse")
            fig_misoa = plot_route(cities, best_route, "Lộ trình MISOA tối ưu")
            st.pyplot(fig_misoa)
            
        with col3:
            st.warning("💰 Hiệu quả kinh tế (1 Chuyến)")
            st.metric("⛽ Xăng tiết kiệm", f"{fuel_saved:.1f} Lít", "Bảo vệ môi trường")
            st.metric("💵 Tiền tiết kiệm", f"{money_saved:,.0f} VNĐ", "Tăng lợi nhuận")
            st.markdown(f"> *Nếu chạy 30 chuyến/tháng, MISOA giúp công ty tiết kiệm **{(money_saved * 30):,.0f} VNĐ** chi phí vận hành!*")
            
        st.markdown("---")
        
        # --- BIỂU ĐỒ SO SÁNH VÀ BẢNG LỘ TRÌNH ---
        col_chart, col_table = st.columns([1.2, 1])
        
        with col_chart:
            st.markdown("### 📉 Đồ thị hội tụ (Convergence Curve)")
            st.caption("So sánh tốc độ tìm đường của MISOA và SOA gốc")
            fig_curve, ax_curve = plt.subplots(figsize=(6, 4))
            
            ax_curve.plot(curve_soa, color='red', linewidth=2, label=f'SOA Gốc (Tốt nhất: {score_soa:.1f} km)')
            ax_curve.plot(curve_misoa, color='blue', linestyle='--', linewidth=2.5, label=f'MISOA (Tốt nhất: {best_score:.1f} km)')
            
            ax_curve.set_xlabel("Số vòng lặp (Iterations)")
            ax_curve.set_ylabel("Tổng quãng đường (km)")
            ax_curve.grid(True, linestyle='--', alpha=0.5)
            ax_curve.legend()
            
            ax_curve.tick_params(colors='#555555')
            for spine in ax_curve.spines.values():
                spine.set_color('#CCCCCC')
                
            st.pyplot(fig_curve)
            
        with col_table:
            st.markdown("### 📋 Bảng kê chi tiết lộ trình")
            st.caption("Danh sách các trạm cần đi theo thứ tự tối ưu")
            
            itinerary = []
            for i in range(len(best_route)):
                current_city = best_route[i]
                next_city = best_route[(i + 1) % len(best_route)]
                dist = np.linalg.norm(cities[current_city] - cities[next_city])
                
                itinerary.append({
                    "Thứ tự": f"Bước {i + 1}",
                    "Trạm": f"Trạm {current_city}",
                    "Khoảng cách chặng (km)": round(dist, 2)
                })
                
            itinerary.append({
                "Thứ tự": "Kết thúc",
                "Trạm": f"Trạm {best_route[0]} (Gốc)",
                "Khoảng cách chặng (km)": "-"
            })
                
            df_itinerary = pd.DataFrame(itinerary)
            st.dataframe(df_itinerary, use_container_width=True, hide_index=True)
            
        st.write("### 🧬 Thứ tự các trạm giao hàng cần đi:")
        route_str = " ➔ ".join([str(x) for x in best_route]) + f" ➔ {best_route[0]}"
        st.code(route_str)

    # ==========================================
    # CHẾ ĐỘ 2: CHẠY NHIỀU LẦN (THỐNG KÊ)
    # ==========================================
    else:
        st.markdown(f"### 🔄 Đang chạy kiểm định {num_runs} lần độc lập. Vui lòng đợi...")
        progress_bar = st.progress(0)
        
        results_soa = []
        results_misoa = []
        
        for i in range(num_runs):
            # Chạy SOA
            score_s, _, _ = soa(search_agents, max_iterations, lower_bound, upper_bound, dimension, objective_function)
            results_soa.append(score_s)
            
            # Chạy MISOA
            score_m, _, _ = misoa(search_agents, max_iterations, lower_bound, upper_bound, dimension, objective_function)
            results_misoa.append(score_m)
            
            progress_bar.progress((i + 1) / num_runs)
            
        st.success("✅ Đã hoàn thành kiểm định thống kê!")
        
        stat_data = {
            "Thuật toán": ["SOA (Bản gốc)", "MISOA (Cải tiến)"],
            "Tốt nhất (Best) ↓": [min(results_soa), min(results_misoa)],
            "Tệ nhất (Worst) ↓": [max(results_soa), max(results_misoa)],
            "Trung bình (Mean) ↓": [np.mean(results_soa), np.mean(results_misoa)],
            "Độ lệch chuẩn (Std) ↓": [np.std(results_soa), np.std(results_misoa)]
        }
        df_stat = pd.DataFrame(stat_data)
        
        for col in ["Tốt nhất (Best) ↓", "Tệ nhất (Worst) ↓", "Trung bình (Mean) ↓", "Độ lệch chuẩn (Std) ↓"]:
            df_stat[col] = df_stat[col].round(2)
            
        st.markdown("### 🏆 Bảng kết quả thống kê tổng hợp")
        st.dataframe(df_stat, use_container_width=True, hide_index=True)
        
        mean_soa = np.mean(results_soa)
        mean_misoa = np.mean(results_misoa)
        
        if mean_misoa < mean_soa:
            st.info(f"🎉 **Kết luận:** Về mặt tổng thể, **MISOA** ổn định và hiệu quả hơn SOA (Trung bình tiết kiệm {mean_soa - mean_misoa:.2f} km/lần chạy).")
        else:
            st.warning(f"⚠️ **Kết luận:** Về mặt tổng thể, **SOA** đang cho kết quả trung bình tốt hơn MISOA trong bài toán TSP tổ hợp này.")
            st.caption("*(Giải thích: Trong một số không gian rời rạc, cơ chế đột biến Levy Flight của MISOA có thể vô tình phá vỡ cấu trúc chuỗi tối ưu (ROV) khiến kết quả trung bình kém hơn thuật toán gốc).*")
        st.markdown("---")
        st.markdown("### 📊 Trực quan hóa kết quả thống kê")

        col_bar, col_box = st.columns(2)

        with col_bar:
            st.markdown("#### 1. Lịch sử các lần chạy")
            st.caption("Biểu đồ so sánh quãng đường của từng lần chạy độc lập")
            
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            x = np.arange(num_runs)
            width = 0.35
            
            ax_bar.bar(x - width/2, results_soa, width, label='SOA Gốc', color='#FF4B4B', alpha=0.8)
            ax_bar.bar(x + width/2, results_misoa, width, label='MISOA', color='#4682B4', alpha=0.9)
            
            ax_bar.set_xlabel("Lần chạy thứ")
            ax_bar.set_ylabel("Tổng quãng đường (km)")
            
            if num_runs <= 20:
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels([f"{i+1}" for i in range(num_runs)])
            else:
                ax_bar.set_xticks([])
            ax_bar.legend()
            ax_bar.grid(True, linestyle='--', alpha=0.3)
            
            ax_bar.tick_params(colors='#555555')
            for spine in ax_bar.spines.values():
                spine.set_color('#CCCCCC')
                
            st.pyplot(fig_bar)
            
        with col_box:
            st.markdown("#### 2. Biểu đồ phân phối (Boxplot)")
            st.caption("Đánh giá độ ổn định (Hộp càng hẹp, dữ liệu càng ổn định)")
            
            fig_box, ax_box = plt.subplots(figsize=(6, 4))
            
            bplot = ax_box.boxplot([results_soa, results_misoa], 
                                    labels=['SOA Gốc', 'MISOA'], 
                                    patch_artist=True,
                                    widths=0.5)
            
            colors = ['#FF4B4B', '#4682B4']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            for median in bplot['medians']:
                median.set_color('black')
                median.set_linewidth(2)
                
            ax_box.set_ylabel("Tổng quãng đường (km)")
            ax_box.grid(True, linestyle='--', alpha=0.3)
            
            ax_box.tick_params(colors='#555555')
            for spine in ax_box.spines.values():
                spine.set_color('#CCCCCC')
                
            st.pyplot(fig_box)
else:
    st.info("👈 Hãy chọn chế độ và bấm 'Bắt đầu Tối Ưu Hóa' bên thanh công cụ!")
    st.markdown("### 🗺️ Bản đồ các trạm giao hàng hiện tại (Chưa tối ưu)")
    
    fig_raw = plot_route(cities, naive_route, "Lộ trình ngẫu nhiên sinh ra", color='gray')
    st.pyplot(fig_raw)