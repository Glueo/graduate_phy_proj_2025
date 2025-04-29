# 作者： Glue
# 创建于 2025年04月18日16时38分18秒
# glueo@icloud.com
# 修改：使用Numba CUDA加速,仅供参考

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from numba import cuda, float32, int32, jit

# 设置中文字体支持
try:
    # 尝试设置为微软雅黑或其他中文字体
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 无法设置中文字体，将使用系统默认字体")

# 双极子距离
d = 2.0  # 双极子中两个电荷间距离 (Å)


# cuda
@cuda.jit
def dipole_force_same_kernel(positions, velocities, d_value, dt, steps, r_cut, max_steps, results):
    """
    同种电荷双极子场中粒子轨迹的CUDA核函数
    参数:
    positions: 形状为 (n, 2) 的数组，包含所有粒子的初始位置
    velocities: 形状为 (n, 2) 的数组，包含所有粒子的初始速度
    d_value: 双极子中两个电荷间的距离
    dt: 时间步长
    steps: 最大模拟步数
    r_cut: 散射截止距离
    max_steps: 粒子最大轨道运行步数，超过则认为被捕获
    results: 形状为 (n, 3) 的数组，存储结果 [是否散射完成, 散射角x分量, 散射角y分量]
    """
    # 获取当前线程的索引
    i = cuda.grid(1)

    # 检查是否在有效范围内
    if i < positions.shape[0]:
        # 初始化位置和速度
        r_x = positions[i, 0]
        r_y = positions[i, 1]
        v_x = velocities[i, 0]
        v_y = velocities[i, 1]

        # 双极子两个电荷的位置
        charge1_x = -d_value / 2
        charge1_y = 0.0
        charge2_x = d_value / 2
        charge2_y = 0.0

        # 模拟步骤
        step_count = 0
        completed = False

        while step_count < steps and not completed:
            # 计算当前加速度
            r1_x = r_x - charge1_x
            r1_y = r_y - charge1_y
            r2_x = r_x - charge2_x
            r2_y = r_y - charge2_y

            dist1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
            dist2 = math.sqrt(r2_x * r2_x + r2_y * r2_y)

            # 防止除以零
            if dist1 < 1e-6 or dist2 < 1e-6:
                # 如果太靠近电荷，结束模拟
                break

            # 计算加速度
            a1_x = r1_x / (dist1 ** 3)
            a1_y = r1_y / (dist1 ** 3)
            a2_x = r2_x / (dist2 ** 3)
            a2_y = r2_y / (dist2 ** 3)

            a_x = a1_x + a2_x
            a_y = a1_y + a2_y

            # Verlet 积分
            r_x_new = r_x + v_x * dt + 0.5 * a_x * dt * dt
            r_y_new = r_y + v_y * dt + 0.5 * a_y * dt * dt

            # 计算新位置的加速度
            r1_x = r_x_new - charge1_x
            r1_y = r_y_new - charge1_y
            r2_x = r_x_new - charge2_x
            r2_y = r_y_new - charge2_y

            dist1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
            dist2 = math.sqrt(r2_x * r2_x + r2_y * r2_y)

            if dist1 < 1e-6 or dist2 < 1e-6:
                break

            a1_x_new = r1_x / (dist1 ** 3)
            a1_y_new = r1_y / (dist1 ** 3)
            a2_x_new = r2_x / (dist2 ** 3)
            a2_y_new = r2_y / (dist2 ** 3)

            a_x_new = a1_x_new + a2_x_new
            a_y_new = a1_y_new + a2_y_new

            # 更新速度
            v_x = v_x + 0.5 * (a_x + a_x_new) * dt
            v_y = v_y + 0.5 * (a_y + a_y_new) * dt

            # 更新位置
            r_x = r_x_new
            r_y = r_y_new

            # 检查是否达到截止条件
            r_norm = math.sqrt(r_x * r_x + r_y * r_y)

            if r_norm > r_cut:
                if step_count < max_steps:  # 不是被捕获的情况
                    # 计算散射角
                    results[i, 0] = 1.0  # 标记为散射完成
                    results[i, 1] = v_x  # 存储速度的x分量
                    results[i, 2] = v_y  # 存储速度的y分量
                    completed = True
                break

            step_count += 1

            # 如果步数超过阈值，认为被捕获
            if step_count >= max_steps:
                break


@cuda.jit
def dipole_force_opposite_kernel(positions, velocities, d_value, dt, steps, r_cut, max_steps, results):
    # 获取当前线程的索引
    i = cuda.grid(1)

    # 检查是否在有效范围内
    if i < positions.shape[0]:
        # 初始化位置和速度
        r_x = positions[i, 0]
        r_y = positions[i, 1]
        v_x = velocities[i, 0]
        v_y = velocities[i, 1]

        # 双极子两个电荷的位置（异种电荷情况）
        charge1_x = d_value / 2  # 正电荷
        charge1_y = 0.0
        charge2_x = -d_value / 2  # 负电荷
        charge2_y = 0.0

        # 模拟步骤
        step_count = 0
        completed = False

        while step_count < steps and not completed:
            # 计算当前加速度
            r1_x = r_x - charge1_x
            r1_y = r_y - charge1_y
            r2_x = r_x - charge2_x
            r2_y = r_y - charge2_y

            dist1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
            dist2 = math.sqrt(r2_x * r2_x + r2_y * r2_y)

            # 防止除以零
            if dist1 < 1e-6 or dist2 < 1e-6:
                break

            # 计算加速度（异种电荷）
            a1_x = r1_x / (dist1 ** 3)
            a1_y = r1_y / (dist1 ** 3)
            a2_x = -r2_x / (dist2 ** 3)  # 负号表示负电荷的吸引力
            a2_y = -r2_y / (dist2 ** 3)

            a_x = a1_x + a2_x
            a_y = a1_y + a2_y

            # Verlet 积分
            r_x_new = r_x + v_x * dt + 0.5 * a_x * dt * dt
            r_y_new = r_y + v_y * dt + 0.5 * a_y * dt * dt

            # 计算新位置的加速度
            r1_x = r_x_new - charge1_x
            r1_y = r_y_new - charge1_y
            r2_x = r_x_new - charge2_x
            r2_y = r_y_new - charge2_y

            dist1 = math.sqrt(r1_x * r1_x + r1_y * r1_y)
            dist2 = math.sqrt(r2_x * r2_x + r2_y * r2_y)

            if dist1 < 1e-6 or dist2 < 1e-6:
                break

            a1_x_new = r1_x / (dist1 ** 3)
            a1_y_new = r1_y / (dist1 ** 3)
            a2_x_new = -r2_x / (dist2 ** 3)
            a2_y_new = -r2_y / (dist2 ** 3)

            a_x_new = a1_x_new + a2_x_new
            a_y_new = a1_y_new + a2_y_new

            # 更新速度
            v_x = v_x + 0.5 * (a_x + a_x_new) * dt
            v_y = v_y + 0.5 * (a_y + a_y_new) * dt

            # 更新位置
            r_x = r_x_new
            r_y = r_y_new

            # 检查是否达到截止条件
            r_norm = math.sqrt(r_x * r_x + r_y * r_y)

            if r_norm > r_cut:
                if step_count < max_steps:  # 不是被捕获的情况
                    # 计算散射角
                    results[i, 0] = 1.0  # 标记为散射完成
                    results[i, 1] = v_x  # 存储速度的x分量
                    results[i, 2] = v_y  # 存储速度的y分量
                    completed = True
                break

            step_count += 1

            # 如果步数超过阈值，认为被捕获
            if step_count >= max_steps:
                break


def run_simulations_same_charges_gpu():
    """使用GPU执行同种电荷散射模拟"""
    num_speed = 500000  # 模拟粒子数
    speed_min, speed_max = 0.5, 2.5

    # 为所有粒子准备初始条件
    speeds = np.random.uniform(speed_min, speed_max, num_speed)
    positions = np.zeros((num_speed, 2), dtype=np.float32)
    velocities = np.zeros((num_speed, 2), dtype=np.float32)

    positions[:, 0] = x0  # 所有粒子起始x坐标
    positions[:, 1] = 0.5  # 所有粒子起始y坐标
    velocities[:, 0] = speeds  # 所有粒子速度x分量
    # y分量都是0（水平入射）

    # 准备结果数组：[是否完成散射, vx, vy]
    results = np.zeros((num_speed, 3), dtype=np.float32)

    # 将数据转移到GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_results = cuda.to_device(results)

    # 设置线程数和块数
    threads_per_block = 256 # 设置线程数
    blocks_per_grid = (num_speed + threads_per_block - 1) // threads_per_block

    # 执行核函数
    print("运行同种电荷-不同速度模拟...")
    dipole_force_same_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d, dt, steps, R_cut, max_orbit_time, d_results
    )

    # 将结果复制回主机
    h_results = d_results.copy_to_host()

    # 处理结果，计算散射角
    scattering_angles_speed = []
    for i in range(num_speed):
        if h_results[i, 0] > 0:  # 散射完成
            vx, vy = h_results[i, 1], h_results[i, 2]
            angle = np.degrees(np.arctan2(vy, vx))
            scattering_angles_speed.append(angle)

    num_angle = 500000
    incident_angle_min, incident_angle_max = -15.0, 15.0

    # 准备初始条件
    angles = np.random.uniform(incident_angle_min, incident_angle_max, num_angle)
    angles_rad = np.radians(angles)

    positions = np.zeros((num_angle, 2), dtype=np.float32)
    velocities = np.zeros((num_angle, 2), dtype=np.float32)

    positions[:, 0] = x0  # 所有粒子起始x坐标
    velocities[:, 0] = 1.5 * np.cos(angles_rad)  # 速度x分量
    velocities[:, 1] = 1.5 * np.sin(angles_rad)  # 速度y分量

    # 准备结果数组
    results = np.zeros((num_angle, 3), dtype=np.float32)

    # 将数据转移到GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_results = cuda.to_device(results)

    # 设置线程数和块数
    blocks_per_grid = (num_angle + threads_per_block - 1) // threads_per_block

    # 执行核函数
    print("运行同种电荷-不同角度模拟...")
    dipole_force_same_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d, dt, steps, R_cut, max_orbit_time, d_results
    )

    # 将结果复制回主机
    h_results = d_results.copy_to_host()

    # 处理结果，计算散射角
    scattering_angles_angle = []
    for i in range(num_angle):
        if h_results[i, 0] > 0:  # 散射完成
            vx, vy = h_results[i, 1], h_results[i, 2]
            angle = np.degrees(np.arctan2(vy, vx))
            scattering_angles_angle.append(angle)

    num_b = 500000
    b_min, b_max = -1.5, 1.5

    # 准备初始条件
    bs = np.random.uniform(b_min, b_max, num_b)

    positions = np.zeros((num_b, 2), dtype=np.float32)
    velocities = np.zeros((num_b, 2), dtype=np.float32)

    positions[:, 0] = x0  # 所有粒子起始x坐标
    positions[:, 1] = bs  # 所有粒子起始y坐标
    velocities[:, 0] = 1.5  # 所有粒子速度x分量为1（水平入射）

    # 准备结果数组
    results = np.zeros((num_b, 3), dtype=np.float32)

    # 将数据转移到GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_results = cuda.to_device(results)

    # 设置线程数和块数
    blocks_per_grid = (num_b + threads_per_block - 1) // threads_per_block

    # 执行核函数
    print("运行同种电荷-不同瞄准距离模拟...")
    dipole_force_same_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d, dt, steps, R_cut, max_orbit_time, d_results
    )

    # 将结果复制回主机
    h_results = d_results.copy_to_host()

    # 处理结果，计算散射角
    scattering_angles_b = []
    for i in range(num_b):
        if h_results[i, 0] > 0:  # 散射完成
            vx, vy = h_results[i, 1], h_results[i, 2]
            angle = np.degrees(np.arctan2(vy, vx))
            scattering_angles_b.append(angle)

    return scattering_angles_speed, scattering_angles_angle, scattering_angles_b


def run_simulations_opposite_charges_gpu():
    num_speed = 500000
    speed_min, speed_max = 0.5, 2.5

    # 为所有粒子准备初始条件
    speeds = np.random.uniform(speed_min, speed_max, num_speed)
    positions = np.zeros((num_speed, 2), dtype=np.float32)
    velocities = np.zeros((num_speed, 2), dtype=np.float32)

    positions[:, 0] = x0  # 所有粒子起始x坐标
    positions[:, 1] = 0.5  # 所有粒子起始y坐标
    velocities[:, 0] = speeds  # 所有粒子速度x分量

    # 准备结果数组
    results = np.zeros((num_speed, 3), dtype=np.float32)

    # 将数据转移到GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_results = cuda.to_device(results)

    # 设置线程数和块数
    threads_per_block = 256
    blocks_per_grid = (num_speed + threads_per_block - 1) // threads_per_block

    # 执行核函数
    print("运行异种电荷-不同速度模拟...")
    dipole_force_opposite_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d, dt, steps, R_cut, max_orbit_time, d_results
    )

    # 将结果复制回主机
    h_results = d_results.copy_to_host()

    # 处理结果，计算散射角
    scattering_angles_speed = []
    for i in range(num_speed):
        if h_results[i, 0] > 0:  # 散射完成
            vx, vy = h_results[i, 1], h_results[i, 2]
            angle = np.degrees(np.arctan2(vy, vx))
            scattering_angles_speed.append(angle)

    num_angle = 500000
    incident_angle_min, incident_angle_max = -30.0, 30.0

    # 准备初始条件
    angles = np.random.uniform(incident_angle_min, incident_angle_max, num_angle)
    angles_rad = np.radians(angles)

    positions = np.zeros((num_angle, 2), dtype=np.float32)
    velocities = np.zeros((num_angle, 2), dtype=np.float32)

    positions[:, 0] = x0
    velocities[:, 0] = 1.5 * np.cos(angles_rad)
    velocities[:, 1] = 1.5 * np.sin(angles_rad)

    # 准备结果数组
    results = np.zeros((num_angle, 3), dtype=np.float32)

    # 将数据转移到GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_results = cuda.to_device(results)

    # 设置线程数和块数
    blocks_per_grid = (num_angle + threads_per_block - 1) // threads_per_block

    # 执行核函数
    print("运行异种电荷-不同角度模拟...")
    dipole_force_opposite_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d, dt, steps, R_cut, max_orbit_time, d_results
    )

    # 将结果复制回主机
    h_results = d_results.copy_to_host()

    # 处理结果，计算散射角
    scattering_angles_angle = []
    for i in range(num_angle):
        if h_results[i, 0] > 0:
            vx, vy = h_results[i, 1], h_results[i, 2]
            angle = np.degrees(np.arctan2(vy, vx))
            scattering_angles_angle.append(angle)

    num_b = 500000
    b_min, b_max = -1.5, 1.5

    # 准备初始条件
    bs = np.random.uniform(b_min, b_max, num_b)

    positions = np.zeros((num_b, 2), dtype=np.float32)
    velocities = np.zeros((num_b, 2), dtype=np.float32)

    positions[:, 0] = x0
    positions[:, 1] = bs
    velocities[:, 0] = 1.5

    # 准备结果数组
    results = np.zeros((num_b, 3), dtype=np.float32)

    # 将数据转移到GPU
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_results = cuda.to_device(results)

    # 设置线程数和块数
    blocks_per_grid = (num_b + threads_per_block - 1) // threads_per_block

    # 执行核函数
    print("运行异种电荷-不同瞄准距离模拟...")
    dipole_force_opposite_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d, dt, steps, R_cut, max_orbit_time, d_results
    )

    # 将结果复制回主机
    h_results = d_results.copy_to_host()

    # 处理结果，计算散射角
    scattering_angles_b = []
    for i in range(num_b):
        if h_results[i, 0] > 0:
            vx, vy = h_results[i, 1], h_results[i, 2]
            angle = np.degrees(np.arctan2(vy, vx))
            scattering_angles_b.append(angle)

    return scattering_angles_speed, scattering_angles_angle, scattering_angles_b


steps = 2000  # 模拟步数
dt = 0.01  # 时间步长
x0 = -4.0  # 入射起始位置
R_cut = 20.0  # 当粒子运动到距离散射中心大于 R_cut 时，认为散射过程结束
max_orbit_time = 2000  # 粒子最大轨道运行时间，超过则认为被捕获


def main():
    # 创建缓存目录
    if not os.path.exists("./cache"):
        os.makedirs("./cache")

    # 创建输出目录
    if not os.path.exists("./output"):
        os.makedirs("./output")

    # 检查CUDA可用性
    if not cuda.is_available():
        print("警告: CUDA不可用，请检查您的GPU驱动和Numba安装")
        return

    # 运行模拟或加载缓存
    try:
        data = np.load("./cache/dipole_scattering_angles_gpu.npz")
        same_speed = data['same_speed']
        same_angle = data['same_angle']
        same_b = data['same_b']
        opposite_speed = data['opposite_speed']
        opposite_angle = data['opposite_angle']
        opposite_b = data['opposite_b']
        print("已加载缓存数据")
    except FileNotFoundError:
        print("未找到缓存，开始运行GPU加速模拟...")

        # 运行同种电荷模拟
        print("开始同种电荷散射模拟...")
        same_speed, same_angle, same_b = run_simulations_same_charges_gpu()

        # 运行异种电荷模拟
        print("开始异种电荷散射模拟...")
        opposite_speed, opposite_angle, opposite_b = run_simulations_opposite_charges_gpu()

        # 保存结果
        np.savez("./cache/dipole_scattering_angles_gpu.npz",
                 same_speed=same_speed,
                 same_angle=same_angle,
                 same_b=same_b,
                 opposite_speed=opposite_speed,
                 opposite_angle=opposite_angle,
                 opposite_b=opposite_b)
        print("GPU模拟完成并已保存缓存")

    fig_same, axes_same = plt.subplots(3, 1, figsize=(10, 16), dpi=300, sharex=True)
    bins = np.linspace(-180, 180, 121)  # 散射角范围（度）

    axes_same[0].hist(same_speed, bins=bins, density=True, color='steelblue', edgecolor='black', alpha=0.7)
    axes_same[0].set_title("同种电荷：不同入射速度下的散射角分布", fontsize=14)
    axes_same[0].set_ylabel("概率密度 (per°)", fontsize=12)

    axes_same[1].hist(same_angle, bins=bins, density=True, color='seagreen', edgecolor='black', alpha=0.7)
    axes_same[1].set_title("同种电荷：不同入射角度下的散射角分布", fontsize=14)
    axes_same[1].set_ylabel("概率密度 (per°)", fontsize=12)

    axes_same[2].hist(same_b, bins=bins, density=True, color='orangered', edgecolor='black', alpha=0.7)
    axes_same[2].set_title("同种电荷：不同瞄准距离下的散射角分布", fontsize=14)
    axes_same[2].set_xlabel("散射角 (度)", fontsize=12)
    axes_same[2].set_ylabel("概率密度 (per°)", fontsize=12)

    plt.tight_layout()
    plt.savefig("./output/Dipole_scattering_same_charges.png", dpi=300)
    plt.close()

    fig_opposite, axes_opposite = plt.subplots(3, 1, figsize=(10, 16), dpi=300, sharex=True)

    axes_opposite[0].hist(opposite_speed, bins=bins, density=True, color='steelblue', edgecolor='black', alpha=0.7)
    axes_opposite[0].set_title("异种电荷：不同入射速度下的散射角分布", fontsize=14)
    axes_opposite[0].set_ylabel("概率密度 (per°)", fontsize=12)

    axes_opposite[1].hist(opposite_angle, bins=bins, density=True, color='seagreen', edgecolor='black', alpha=0.7)
    axes_opposite[1].set_title("异种电荷：不同入射角度下的散射角分布", fontsize=14)
    axes_opposite[1].set_ylabel("概率密度 (per°)", fontsize=12)

    axes_opposite[2].hist(opposite_b, bins=bins, density=True, color='orangered', edgecolor='black', alpha=0.7)
    axes_opposite[2].set_title("异种电荷：不同瞄准距离下的散射角分布", fontsize=14)
    axes_opposite[2].set_xlabel("散射角 (度)", fontsize=12)
    axes_opposite[2].set_ylabel("概率密度 (per°)", fontsize=12)

    plt.tight_layout()
    plt.savefig("./output/Dipole_scattering_opposite_charges.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
