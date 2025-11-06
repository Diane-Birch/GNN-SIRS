import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus, Sigmoid, Parameter
import torch.distributions as dist


class OakDiseaseSIRSModel(MessagePassing):
    def __init__(self, n_features, delta_t, aggr='add'):
        super(OakDiseaseSIRSModel, self).__init__(aggr=aggr, flow='source_to_target')

        # 可学习参数
        self.beta = Parameter(torch.tensor(0.1))  # 感染率参数
        self.L = Parameter(torch.tensor(1.0))  # 距离衰减参数
        self.gamma = Parameter(torch.tensor(0.05))  # 恢复率参数
        self.xi = Parameter(torch.tensor(0.01))  # 免疫力丧失率参数

        # 随机性参数
        self.infection_noise = Parameter(torch.tensor(0.01))
        self.recovery_noise = Parameter(torch.tensor(0.01))

        self.delta_t = delta_t

    def forward(self, x, edge_index, edge_attr=None, global_recovery_target=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr,
                              global_recovery_target=global_recovery_target)

    def message(self, x_i, x_j, edge_attr):
        """
        x_i: 目标节点特征 [S,I,R, 感染时间, 免疫力因子, 人为干预强度, x, y...]
        x_j: 源节点特征
        """
        # 提取状态信息
        S_i, I_i, R_i = x_i[:, 0:1], x_i[:, 1:2], x_i[:, 2:3]
        S_j, I_j, R_j = x_j[:, 0:1], x_j[:, 1:2], x_j[:, 2:3]

        # 只有易感节点(S)和感染节点(I)之间才可能传播
        susceptible_mask = S_i
        infected_mask = I_j

        # 计算距离
        if edge_attr is not None:
            distance = edge_attr[:, 0:1]
        else:
            # 从节点坐标计算距离
            pos_i = x_i[:, 6:8]  # 假设6-7列是坐标
            pos_j = x_j[:, 6:8]
            distance = torch.norm(pos_i - pos_j, dim=1, keepdim=True)

        # 使用参数化模型计算感染贡献
        # lambda_i = sum_{j in infected neighbors} beta * exp(-d_ij / L)
        infection_contribution = self.beta * torch.exp(-distance / self.L)

        # 应用状态掩码
        infection_contribution = infection_contribution * susceptible_mask * infected_mask

        return infection_contribution

    def update(self, aggr_out, x, global_recovery_target=None):
        """
        更新节点状态，使用SIRS微分方程模型
        dS/dt = -β * I * S / N + ξ * R
        dI/dt = β * I * S / N - γ * I
        dR/dt = γ * I - ξ * R
        """
        # 提取当前状态
        S, I, R = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        infection_time = x[:, 3:4]  # 感染时间
        immunity_factor = x[:, 4:5]  # 免疫力因子
        intervention = x[:, 5:6]  # 人为干预强度
        other_features = x[:, 6:]  # 其他特征(坐标等)

        # 计算总感染率 (来自邻居的贡献)
        # 这里aggr_out已经是lambda_i = sum_{j in infected neighbors} beta * exp(-d_ij / L)
        # 在SIRS模型中，感染项是 β * I * S / N
        # 由于我们是在节点级别建模，每个节点有自己的状态，所以N=1
        # 因此感染项 = lambda_i * S
        infection_term = aggr_out * S

        # 考虑免疫力影响
        effective_infection_term = infection_term * (1 - immunity_factor)

        # 计算恢复项 - 考虑人为干预影响
        recovery_term = self.gamma * I * intervention

        # 计算免疫力丧失项
        immunity_loss_term = self.xi * R

        # 使用欧拉方法更新SIRS状态
        dS_dt = -effective_infection_term + immunity_loss_term
        dI_dt = effective_infection_term - recovery_term
        dR_dt = recovery_term - immunity_loss_term

        # 更新状态
        S_new = S + dS_dt * self.delta_t
        I_new = I + dI_dt * self.delta_t
        R_new = R + dR_dt * self.delta_t

        # 确保状态非负
        S_new = torch.clamp(S_new, 0, 1)
        I_new = torch.clamp(I_new, 0, 1)
        R_new = torch.clamp(R_new, 0, 1)

        # 归一化，确保S+I+R=1
        total = S_new + I_new + R_new
        S_new = S_new / total
        I_new = I_new / total
        R_new = R_new / total

        # 更新感染时间
        # 如果从S或R变为I，则重置感染时间
        # 如果保持I状态，则增加时间
        new_infection = (I < 0.5) & (I_new > 0.5)  # 新感染
        infection_time_new = torch.where(
            new_infection,
            torch.ones_like(infection_time) * self.delta_t,
            torch.where(
                I > 0.5,  # 保持感染
                infection_time + self.delta_t,
                infection_time  # 其他情况保持不变
            )
        )

        # 计算新感染和恢复的数量 (用于损失函数)
        new_infections_count = torch.sum((I < 0.5) & (I_new > 0.5)).float()
        new_recoveries_count = torch.sum((I > 0.5) & (I_new < 0.5)).float()

        # 组合新状态
        updated_x = torch.cat([
            S_new, I_new, R_new,
            infection_time_new,
            immunity_factor,
            intervention,
            other_features
        ], dim=1)

        return updated_x, effective_infection_term, recovery_term, new_infections_count, new_recoveries_count


class OakDiseaseSIRSystem(OakDiseaseSIRSModel):
    def __init__(self, n_features, delta_t, edge_index):
        super(OakDiseaseSIRSystem, self).__init__(n_features, delta_t)
        self.delta_t = delta_t
        self.edge_index = edge_index

    def forward(self, graph_data, global_recovery_target=None):
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = getattr(graph_data, 'edge_attr', None)

        updated_x, infection_term, recovery_term, new_infections, new_recoveries = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, global_recovery_target=global_recovery_target
        )

        return updated_x, infection_term, recovery_term, new_infections, new_recoveries

    def loss(self, graph_data, global_recovery_target=None, infection_time_penalty=1.0):
        """
        计算损失函数 - 匹配已知感染时间和恢复数量
        """
        updated_x, infection_term, recovery_term, new_infections, new_recoveries = self.forward(
            graph_data, global_recovery_target=global_recovery_target
        )

        # 提取真实标签
        true_state = graph_data.y[:, 0:3]  # 真实状态 [S,I,R]
        true_infection_time = graph_data.y[:, 3:4]  # 真实感染时间

        # 状态损失
        state_loss = F.mse_loss(updated_x[:, 0:3], true_state)

        # 感染时间损失 (只对感染节点)
        infected_mask = (true_state[:, 1] > 0.5).float().unsqueeze(1)
        time_loss = F.mse_loss(
            updated_x[:, 3:4] * infected_mask,
            true_infection_time * infected_mask
        )

        # 恢复数量约束损失 (如果提供了目标)
        recovery_loss = 0
        if global_recovery_target is not None:
            recovery_loss = F.mse_loss(
                new_recoveries.unsqueeze(0),
                torch.tensor([global_recovery_target],
                             dtype=torch.float32,
                             device=updated_x.device)
            )

        # 感染时间惩罚 - 确保未感染节点感染时间为0
        not_infected_mask = (true_state[:, 1] < 0.5).float().unsqueeze(1)
        infection_time_penalty_loss = torch.mean(
            (updated_x[:, 3:4] * not_infected_mask) ** 2
        )

        # 参数正则化 - 防止参数过大
        param_regularization = (
                self.beta ** 2 + self.L ** 2 + self.gamma ** 2 + self.xi ** 2 +
                self.infection_noise ** 2 + self.recovery_noise ** 2
        )

        total_loss = (state_loss * 10.0 +
                      time_loss * 5.0 +
                      recovery_loss * 2.0 +
                      infection_time_penalty_loss * infection_time_penalty +
                      param_regularization * 0.01)

        return total_loss, {
            'state_loss': state_loss.item(),
            'time_loss': time_loss.item(),
            'recovery_loss': recovery_loss.item() if global_recovery_target else 0,
            'infection_time_penalty': infection_time_penalty_loss.item(),
            'param_regularization': param_regularization.item(),
            'beta': self.beta.item(),
            'L': self.L.item(),
            'gamma': self.gamma.item(),
            'xi': self.xi.item()
        }

    def simulate_epidemic(self, initial_graph, steps, recovery_targets=None):
        """
        模拟疫情传播，考虑逐年恢复目标
        """
        current_graph = initial_graph
        results = []

        for step in range(steps):
            # 获取当前年的恢复目标
            current_recovery_target = recovery_targets[step] if recovery_targets and step < len(
                recovery_targets) else None

            updated_x, infection_term, recovery_term, new_infections, new_recoveries = self.forward(
                current_graph, global_recovery_target=current_recovery_target
            )

            # 更新图数据
            current_graph.x = updated_x.detach()

            # 计算各种统计量
            S_count = torch.sum(updated_x[:, 0] > 0.5).item()
            I_count = torch.sum(updated_x[:, 1] > 0.5).item()
            R_count = torch.sum(updated_x[:, 2] > 0.5).item()

            # 记录结果
            results.append({
                'step': step,
                'S_count': S_count,
                'I_count': I_count,
                'R_count': R_count,
                'new_infections': new_infections.item(),
                'new_recoveries': new_recoveries.item(),
                'infection_term_mean': torch.mean(infection_term).item(),
                'recovery_term_mean': torch.mean(recovery_term).item(),
                'beta': self.beta.item(),
                'L': self.L.item(),
                'gamma': self.gamma.item(),
                'xi': self.xi.item()
            })

        return results


def prepare_oak_sir_data(oak_positions, known_infection_times, immunity_factors=None,
                         intervention_levels=None, infection_threshold=1000):
    """
    准备橡树SIR图数据
    oak_positions: [n_trees, 2] 橡树坐标
    known_infection_times: [n_trees] 已知感染时间(-1表示未感染或未知)
    immunity_factors: [n_trees] 个体免疫因子(0-1)
    intervention_levels: [n_trees] 人为干预强度(0-1)
    """
    n_trees = len(oak_positions)

    # 默认值
    if immunity_factors is None:
        immunity_factors = np.random.uniform(0.1, 0.5, n_trees)  # 免疫力因子
    if intervention_levels is None:
        intervention_levels = np.random.uniform(0.05, 0.3, n_trees)  # 干预强度

    # 节点特征: [S, I, R, 感染时间, 免疫力因子, 干预强度, x坐标, y坐标]
    node_features = []
    for i in range(n_trees):
        if known_infection_times[i] >= 0:  # 已感染
            S, I, R = 0.0, 1.0, 0.0
            infection_time = known_infection_times[i]
        else:  # 未感染
            S, I, R = 1.0, 0.0, 0.0
            infection_time = 0.0

        features = [
            S, I, R,
            infection_time,
            immunity_factors[i],
            intervention_levels[i],
            oak_positions[i, 0],
            oak_positions[i, 1]
        ]
        node_features.append(features)

    # 构建边 (基于距离)
    edge_index = []
    edge_attr = []

    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            distance = np.linalg.norm(oak_positions[i] - oak_positions[j])
            if distance < infection_threshold:
                # 双向边
                edge_index.append([i, j])
                edge_index.append([j, i])
                # 边属性: [距离]
                edge_attr.append([distance])
                edge_attr.append([distance])

    return {
        'x': torch.tensor(node_features, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
        'y': torch.tensor(node_features, dtype=torch.float32)  # 目标
    }


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    n_trees = 200
    oak_positions = np.random.rand(n_trees, 2) * 1000

    # 已知部分树木的首次感染时间
    known_infection_times = -np.ones(n_trees)
    initial_infected = np.random.choice(n_trees, 10, replace=False)  # 随机选择10棵初始感染
    known_infection_times[initial_infected] = 0

    # 已知每年的总恢复数量
    recovery_targets = np.random.randint(5, 20, 100)  # 未来100年每年的恢复目标

    # 准备数据
    graph_data = prepare_oak_sir_data(oak_positions, known_infection_times)

    # 创建模型
    model = OakDiseaseSIRSystem(n_features=8, delta_t=0.1, edge_index=graph_data['edge_index'])

    # 训练参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("开始训练...")
    for epoch in range(500):
        optimizer.zero_grad()

        # 随机选择一年的恢复目标
        recovery_target = np.random.choice(recovery_targets) if epoch % 10 == 0 else None

        loss, loss_components = model.loss(graph_data, global_recovery_target=recovery_target)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"  Components: {loss_components}")

    # 模拟未来传播
    print("\n开始模拟...")
    model.eval()  # 设置为评估模式
    results = model.simulate_epidemic(graph_data, steps=100, recovery_targets=recovery_targets)

    # 输出最终结果
    final_result = results[-1]
    print(
        f"最终状态 - 健康: {final_result['S_count']}, 感染: {final_result['I_count']}, 恢复: {final_result['R_count']}")
    print(
        f"学习到的参数 - beta: {final_result['beta']:.4f}, L: {final_result['L']:.4f}, gamma: {final_result['gamma']:.4f}, xi: {final_result['xi']:.4f}")

    # 绘制传播曲线
    import matplotlib.pyplot as plt

    steps = [r['step'] for r in results]
    S_counts = [r['S_count'] for r in results]
    I_counts = [r['I_count'] for r in results]
    R_counts = [r['R_count'] for r in results]
    new_recoveries = [r['new_recoveries'] for r in results]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(steps, S_counts, label='易感橡树', color='green')
    plt.plot(steps, I_counts, label='感染橡树', color='red')
    plt.plot(steps, R_counts, label='恢复橡树', color='blue')
    plt.xlabel('时间步')
    plt.ylabel('橡树数量')
    plt.title('橡树病害SIRS模型传播动态')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(steps, new_recoveries, label='每年恢复数量', color='orange')
    if recovery_targets is not None:
        plt.plot(steps, recovery_targets[:100], label='恢复目标', color='purple', linestyle='--')
    plt.xlabel('时间步')
    plt.ylabel('恢复数量')
    plt.title('恢复数量与目标对比')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()