# Latent VLA JEV 流程图

这份文档包含三张可编辑的 Mermaid 流程图：

1. 在线推理主流程：VLM 在子任务边界调用，Decision JEV 在子任务内部循环控制。
2. Stage 1 训练流程：Latent World Model 预测子任务终点潜特征。
3. Stage 2 训练流程：Decision JEV 根据当前状态、终点潜特征和 PT 预测动作、执行后 PT，并判断当前观测是否完成（FT）。

对应的 Mermaid 源文件位于 [`flowcharts/`](flowcharts/)。

## 1. 在线推理主流程

```mermaid
flowchart TB
    subgraph INIT[子任务初始化：每个子任务执行一次]
        TASK[总任务指令] --> VLM[VLM]
        HIST[已完成子任务历史] --> VLM
        IMG0[子任务开始图像 I_start] --> VLM
        VLM --> SUB[当前子任务描述与完成条件]
        SUB --> TE[Text Encoder]
        TE --> C[子任务语义特征 c_j]
        IMG0 --> VE0[Vision Encoder]
        VE0 --> Z0[当前视觉特征 z_start]
        Z0 --> LWM[Latent World Model<br/>子任务终点预测]
        C --> LWM
        LWM --> GOAL[预测终点潜特征<br/>z_hat_terminal_j]
        RESET[PT 初始值为 0] --> PT[当前进度 PT]
    end

    subgraph LOOP[子任务内部闭环：每个控制周期执行一次]
        IMG[当前图像 I_t] --> VE[Vision Encoder]
        VE --> Z[当前视觉特征 z_t]
        STATE[机器人状态 s_t] --> RE[Robot Encoder]
        RE --> R[机器人状态特征 r_t]
        Z --> JEV[Decision JEV]
        GOAL --> JEV
        R --> JEV
        PT --> JEV
        JEV --> ACTION[并行离散 Action Tokens<br/>每个 token=一次完整6D EEPose delta]
        JEV --> PTNEXT[预测执行后的 PT 序列]
        JEV --> FTCURRENT[判断当前观测 FT_t]
        ACTION --> DECODE[动作 Token 解码]
        DECODE --> EE[EEPose Delta]
        EE --> IK[逆运动学 IK]
        IK --> COMMAND[关节控制指令]
        COMMAND --> ROBOT[机器人执行]
        ROBOT --> OBS[获得下一帧图像与状态]
        OBS --> IMG
        OBS --> STATE
        PTNEXT --> PTUPDATE[仅对实际执行的动作前缀更新 PT]
        PTUPDATE --> PT
        FTCURRENT --> GATE{当前 FT 为 1?}
    end

    GATE -- 否：继续当前子任务 --> JEV
    GATE -- 是：当前子任务完成 --> UPDATE[记录已完成子任务<br/>重置 PT = 0]
    UPDATE --> HIST
    UPDATE --> VLM
    UPDATE --> IMG0
    UPDATE --> RESET
```

## 2. Stage 1：Latent World Model 训练流程

```mermaid
flowchart LR
    subgraph DATA[子任务片段训练数据]
        CURRENT[当前帧 I_t]
        TEXT[子任务描述 q_j]
        END[子任务最后一帧 I_end]
    end
    CURRENT --> VE[Vision Encoder<br/>共享或冻结]
    VE --> Z[当前视觉特征 z_t]
    TEXT --> TE[Text Encoder]
    TE --> C[子任务语义特征 c_j]
    Z --> LWM[Latent World Model]
    C --> LWM
    LWM --> PRED[预测子任务终点特征<br/>z_hat_terminal]
    END --> VEGT[Vision Encoder<br/>同一特征空间]
    VEGT --> GT[真实子任务终点特征<br/>z_terminal]
    PRED --> LOSS[终点特征损失]
    GT --> LOSS
```

## 3. Stage 2：Decision JEV 训练流程

```mermaid
flowchart LR
    subgraph INPUT[子任务内训练样本]
        IMAGE[当前图像 I_t]
        STATE[机器人状态 s_t]
        TEXT[子任务描述 q_j]
        PTLABEL[当前进度标签<br/>PT 为 k / K]
    end
    IMAGE --> VE[Vision Encoder]
    VE --> Z[当前视觉特征 z_t]
    TEXT --> TE[Text Encoder]
    TE --> C[子任务语义特征 c_j]
    Z --> LWM[预训练 Latent World Model]
    C --> LWM
    LWM --> GOAL[预测终点潜特征<br/>z_hat_terminal]
    STATE --> RE[Robot Encoder]
    RE --> R[机器人状态特征 r_t]
    Z --> JEV[Decision JEV]
    GOAL --> JEV
    R --> JEV
    PTLABEL --> JEV
    JEV --> ACTION[预测并行 Action Tokens<br/>每个 token=完整6D EEPose delta]
    JEV --> PREDPT[预测执行后的 PT 序列]
    JEV --> PREDFT[判断当前观测 FT_t]
    ACTION --> ACTIONGT[真实 Action Token]
    PREDPT --> PTGT[真实执行后 PT]
    PREDFT --> FTGT[真实当前状态 FT_t]
    ACTIONGT --> LOSS[联合损失]
    PTGT --> LOSS
    FTGT --> LOSS
```
