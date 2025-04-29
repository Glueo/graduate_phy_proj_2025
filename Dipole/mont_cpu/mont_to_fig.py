# 作者： Glue
# 创建于 2025年04月18日16时38分18秒
# glueo@icloud.com

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# 设置中文字体支持
try:
    # 尝试设置为微软雅黑或其他中文字体
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 无法设置中文字体，将使用系统默认字体")

# 双极子距离
d = 2.0  # 双极子中两个电荷间距离


# 势能函数定义
# 双极子势能函数（同种电荷）
def dipole_potential_same(r_vec, k=1.0):
    """
    双极子势能（同种电荷），粒子与两个同号电荷的库仑相互作用
    r_vec: 粒子位置向量 [x, y]
    k: 库仑常数 (取为1.0作为单位简化)
    """
    # 双极子两个电荷的位置
    charge1_pos = np.array([-d / 2, 0])
    charge2_pos = np.array([d / 2, 0])

    # 计算到两个电荷的距离
    r1 = r_vec - charge1_pos
    r2 = r_vec - charge2_pos

    dist1 = np.linalg.norm(r1)
    dist2 = np.linalg.norm(r2)

    # 防止除以零
    if dist1 < 1e-6 or dist2 < 1e-6:
        return 1e6  # 非常大的势能值作为排斥

    # 同种电荷：两个正电荷
    potential = k * (1 / dist1 + 1 / dist2)

    return potential


# 双极子势能函数（异种电荷）
def dipole_potential_opposite(r_vec, k=1.0):
    """
    双极子势能（异种电荷），粒子与一正一负电荷的库仑相互作用
    r_vec: 粒子位置向量 [x, y]
    k: 库仑常数 (取为1.0作为单位简化)
    """
    # 双极子两个电荷的位置
    charge1_pos = np.array([d / 2, 0])  # 正电荷
    charge2_pos = np.array([-d / 2, 0])  # 负电荷

    # 计算到两个电荷的距离
    r1 = r_vec - charge1_pos
    r2 = r_vec - charge2_pos

    dist1 = np.linalg.norm(r1)
    dist2 = np.linalg.norm(r2)

    # 防止除以零
    if dist1 < 1e-6 or dist2 < 1e-6:
        return 1e6  # 非常大的势能值作为排斥

    # 异种电荷：一正一负
    potential = k * (1 / dist1 - 1 / dist2)

    return potential


# 计算双极子势能的力（同种电荷）
def dipole_force_same(r_vec, k=1.0):
    """计算在同种电荷双极子场中粒子受到的力"""
    # 双极子两个电荷的位置
    charge1_pos = np.array([-d / 2, 0])
    charge2_pos = np.array([d / 2, 0])

    # 计算到两个电荷的距离向量
    r1 = r_vec - charge1_pos
    r2 = r_vec - charge2_pos

    dist1 = np.linalg.norm(r1)
    dist2 = np.linalg.norm(r2)

    # 防止除以零
    if dist1 < 1e-6 or dist2 < 1e-6:
        return np.array([0.0, 0.0])  # 如果太靠近电荷，返回零力

    # 库仑力 F = -k*q1*q2/r^2 * (r_vec/r)
    # 同种电荷情况下，两个力都是排斥力
    force1 = k * r1 / (dist1 ** 3)
    force2 = k * r2 / (dist2 ** 3)

    return force1 + force2


# 计算双极子势能的力（异种电荷）
def dipole_force_opposite(r_vec, k=1.0):
    """计算在异种电荷双极子场中粒子受到的力"""
    # 双极子两个电荷的位置
    charge1_pos = np.array([d / 2, 0])  # 正电荷
    charge2_pos = np.array([-d / 2, 0])  # 负电荷

    # 计算到两个电荷的距离向量
    r1 = r_vec - charge1_pos
    r2 = r_vec - charge2_pos

    dist1 = np.linalg.norm(r1)
    dist2 = np.linalg.norm(r2)

    # 防止除以零
    if dist1 < 1e-6 or dist2 < 1e-6:
        return np.array([0.0, 0.0])  # 如果太靠近电荷，返回零力

    # 库仑力 F = -k*q1*q2/r^2 * (r_vec/r)
    # 异种电荷情况下，第一个是排斥力，第二个是吸引力
    force1 = k * r1 / (dist1 ** 3)
    force2 = -k * r2 / (dist2 ** 3)  # 负号表示负电荷的吸引力

    return force1 + force2


#configs
steps = 2000  # 模拟步数
dt = 0.01  # 时间步长
x0 = -4.0  # 入射起始位置

R_cut = 20.0  # 当粒子运动到距离散射中心大于 R_cut 时，认为散射过程结束
max_orbit_time = 2000  # 粒子最大轨道运行时间，超过则认为被捕获


def is_captured(trajectory):
    """检查粒子是否被捕获（绕圈）"""
    if len(trajectory) < max_orbit_time:
        return False

    # 检查粒子是否长时间未能逃逸
    return True

# 同种电荷散射模拟
def run_simulations_same_charges():
    num_speed = 20000  # 模拟粒子数
    speed_min, speed_max = 0.1, 2.1
    scattering_angles_speed = []  # 存储散射角（单位：度）

    for _ in tqdm(range(num_speed), desc="同种电荷-不同速度"):
        # 固定瞄准距离 b = 0.5，入射角 0°
        r = np.array([x0, 0.5])
        # 随机采样速度
        speed = np.random.uniform(speed_min, speed_max)
        v = np.array([speed, 0.0])

        # 记录轨迹用于检测捕获
        trajectory = [r.copy()]

        # Verlet 积分模拟
        for i in range(steps):
            a_vec = dipole_force_same(r)
            r = r + v * dt + 0.5 * a_vec * dt ** 2
            a_new = dipole_force_same(r)
            v = v + 0.5 * (a_vec + a_new) * dt

            trajectory.append(r.copy())

            if np.linalg.norm(r) > R_cut:
                # 检查是否为散射（而非捕获）
                if not is_captured(trajectory):
                    angle = np.arctan2(v[1], v[0])  # 以弧度表示
                    scattering_angles_speed.append(np.degrees(angle))
                break

            # 如果粒子运行时间超过阈值，则认为被捕获
            if i >= max_orbit_time:
                break


    num_angle = 20000
    incident_angle_min, incident_angle_max = -30.0, 30.0
    scattering_angles_angle = []  # 存储最终散射角（度）

    for _ in tqdm(range(num_angle), desc="同种电荷-不同角度"):
        r = np.array([x0, 0.0])
        # 随机采样入射角度（以度计），转换为弧度，保证方向仍向右
        theta_inc = np.radians(np.random.uniform(incident_angle_min, incident_angle_max))
        v = 1.0 * np.array([np.cos(theta_inc), np.sin(theta_inc)])

        trajectory = [r.copy()]

        for i in range(steps):
            a_vec = dipole_force_same(r)
            r = r + v * dt + 0.5 * a_vec * dt ** 2
            a_new = dipole_force_same(r)
            v = v + 0.5 * (a_vec + a_new) * dt

            trajectory.append(r.copy())

            if np.linalg.norm(r) > R_cut:
                if not is_captured(trajectory):
                    angle = np.arctan2(v[1], v[0])
                    scattering_angles_angle.append(np.degrees(angle))
                break

            if i >= max_orbit_time:
                break


    num_b = 20000
    b_min, b_max = -1.3, 1.3
    scattering_angles_b = []  # 存储散射角（度）

    for _ in tqdm(range(num_b), desc="同种电荷-不同瞄准距离"):
        # 随机采样瞄准距离 b
        b = np.random.uniform(b_min, b_max)
        r = np.array([x0, b])
        v = np.array([1.0, 0.0])  # 水平入射

        trajectory = [r.copy()]

        for i in range(steps):
            a_vec = dipole_force_same(r)
            r = r + v * dt + 0.5 * a_vec * dt ** 2
            a_new = dipole_force_same(r)
            v = v + 0.5 * (a_vec + a_new) * dt

            trajectory.append(r.copy())

            if np.linalg.norm(r) > R_cut:
                if not is_captured(trajectory):
                    angle = np.arctan2(v[1], v[0])
                    scattering_angles_b.append(np.degrees(angle))
                break

            if i >= max_orbit_time:
                break

    return scattering_angles_speed, scattering_angles_angle, scattering_angles_b



# 异种电荷散射模拟
def run_simulations_opposite_charges():
    # 模拟组1：不同入射速度（异种电荷）
    num_speed = 20000
    speed_min, speed_max = 0.1,2.1
    scattering_angles_speed = []

    for _ in tqdm(range(num_speed), desc="异种电荷-不同速度"):
        r = np.array([x0, 0.5])
        speed = np.random.uniform(speed_min, speed_max)
        v = np.array([speed, 0.0])

        trajectory = [r.copy()]

        for i in range(steps):
            a_vec = dipole_force_opposite(r)
            r = r + v * dt + 0.5 * a_vec * dt ** 2
            a_new = dipole_force_opposite(r)
            v = v + 0.5 * (a_vec + a_new) * dt

            trajectory.append(r.copy())

            if np.linalg.norm(r) > R_cut:
                if not is_captured(trajectory):
                    angle = np.arctan2(v[1], v[0])
                    scattering_angles_speed.append(np.degrees(angle))
                break

            if i >= max_orbit_time:
                break


    num_angle = 20000
    incident_angle_min, incident_angle_max = -30.0, 30.0
    scattering_angles_angle = []

    for _ in tqdm(range(num_angle), desc="异种电荷-不同角度"):
        r = np.array([x0, 0.0])
        theta_inc = np.radians(np.random.uniform(incident_angle_min, incident_angle_max))
        v = 1.0 * np.array([np.cos(theta_inc), np.sin(theta_inc)])

        trajectory = [r.copy()]

        for i in range(steps):
            a_vec = dipole_force_opposite(r)
            r = r + v * dt + 0.5 * a_vec * dt ** 2
            a_new = dipole_force_opposite(r)
            v = v + 0.5 * (a_vec + a_new) * dt

            trajectory.append(r.copy())

            if np.linalg.norm(r) > R_cut:
                if not is_captured(trajectory):
                    angle = np.arctan2(v[1], v[0])
                    scattering_angles_angle.append(np.degrees(angle))
                break

            if i >= max_orbit_time:
                break


    num_b = 20000
    b_min, b_max = -1.3, 1.3
    scattering_angles_b = []

    for _ in tqdm(range(num_b), desc="异种电荷-不同瞄准距离"):
        b = np.random.uniform(b_min, b_max)
        r = np.array([x0, b])
        v = np.array([1.0, 0.0])

        trajectory = [r.copy()]

        for i in range(steps):
            a_vec = dipole_force_opposite(r)
            r = r + v * dt + 0.5 * a_vec * dt ** 2
            a_new = dipole_force_opposite(r)
            v = v + 0.5 * (a_vec + a_new) * dt

            trajectory.append(r.copy())

            if np.linalg.norm(r) > R_cut:
                if not is_captured(trajectory):
                    angle = np.arctan2(v[1], v[0])
                    scattering_angles_b.append(np.degrees(angle))
                break

            if i >= max_orbit_time:
                break

    return scattering_angles_speed, scattering_angles_angle, scattering_angles_b


# 创建缓存目录
if not os.path.exists("./cache"):
    os.makedirs("./cache")

# 创建输出目录
if not os.path.exists("./output"):
    os.makedirs("./output")

# 运行模拟或加载缓存
try:
    data = np.load("./cache/dipole_scattering_angles.npz")
    same_speed = data['same_speed']
    same_angle = data['same_angle']
    same_b = data['same_b']
    opposite_speed = data['opposite_speed']
    opposite_angle = data['opposite_angle']
    opposite_b = data['opposite_b']
    print("已加载缓存数据")
except FileNotFoundError:
    print("未找到缓存，开始运行模拟...")
    # 运行同种电荷模拟
    same_speed, same_angle, same_b = run_simulations_same_charges()

    # 运行异种电荷模拟
    opposite_speed, opposite_angle, opposite_b = run_simulations_opposite_charges()

    # 保存结果
    np.savez("./cache/dipole_scattering_angles.npz",
             same_speed=same_speed,
             same_angle=same_angle,
             same_b=same_b,
             opposite_speed=opposite_speed,
             opposite_angle=opposite_angle,
             opposite_b=opposite_b)
    print("模拟完成并已保存缓存")

# plot1
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

#plot2
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