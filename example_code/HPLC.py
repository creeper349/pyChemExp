import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
L = 10  # 色谱柱长度，单位：cm
d = 0.5  # 色谱柱直径，单位：cm
v = 1  # 流速，单位：cm/min
D = 0.01  # 扩散系数，单位：cm^2/min
Ka = 0.5  # 吸附常数
dt = 0.1  # 时间步长
dx = 0.1  # 空间步长
time_steps = 100  # 模拟时间步数

# 初始化浓度数组
C = np.zeros((time_steps, int(L / dx)))

# 初始条件：样品进入色谱柱
C[0, 0] = 1  # 在起始点浓度为1

# 执行数值计算
for t in range(1, time_steps):
    for x in range(1, int(L / dx) - 1):
        # 使用有限差分法更新浓度
        diffusion_term = D * (C[t-1, x+1] - 2 * C[t-1, x] + C[t-1, x-1]) / dx**2
        convection_term = -v * (C[t-1, x] - C[t-1, x-1]) / dx
        adsorption_term = -Ka * C[t-1, x]
        
        # 更新浓度
        C[t, x] = C[t-1, x] + dt * (diffusion_term + convection_term + adsorption_term)

# 可视化结果
plt.imshow(C, aspect='auto', cmap='viridis', extent=[0, L, 0, time_steps])
plt.colorbar(label="Concentration")
plt.xlabel("Position (cm)")
plt.ylabel("Time (min)")
plt.title("Chromatographic Separation Simulation")
plt.show()
