import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

abun_names = ["A_Saliva","A_Stool","F4_gut","F4_tongue","M3_gut","M3_tongue"]

def calc(name):
    # ================== 1. 读取数据 ==================
    with open("data/{name}_abun.csv".format(name=name), "r") as f:
        lines = f.readlines()

    # 第一行是 header，去掉 "Taxa"
    taxa_names = [line.strip().split(',')[0] for line in lines[1:]]
    data = []
    for line in lines[1:]:
        values = list(map(float, line.strip().split(',')[1:]))
        data.append(values)

    X = np.array(data)  # shape: (n_taxa, n_samples)
    print(X.shape)
    # ================== 2. CLR 变换（抗组成性偏差）==================
    def clr_transform(mat):
        mat = mat + 1e-6  # 防止 log(0)
        log_mat = np.log(mat)
        centered = log_mat - log_mat.mean(axis=1, keepdims=True)
        return centered

    X_clr = clr_transform(X)

    # ================== 3. 计算 Spearman 相关矩阵和 p-value 矩阵 ==================
    n_taxa = X_clr.shape[0]
    corr_matrix = np.zeros((n_taxa, n_taxa))
    pval_matrix = np.zeros((n_taxa, n_taxa))

    for i in range(n_taxa):
        for j in range(i+1, n_taxa):
            rho, pval = spearmanr(X_clr[i, :], X_clr[j, :])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval

    # 对角线设为 0（或 NaN），但这里保持为 0
    np.fill_diagonal(pval_matrix, 1.0)

    # ================== 4. 多重检验校正 (FDR-BH) ==================
    pvals_flat = pval_matrix.flatten()
    _, pvals_corrected, _, _ = multipletests(pvals_flat, method='fdr_bh')
    pvals_corrected = pvals_corrected.reshape(pval_matrix.shape)

    # ================== 5. 构建稀疏邻接矩阵（adjacency matrix）==================
    # 仅保留 FDR < 0.05 且 |rho| > 0.3 的边（可调阈值）
    adj_matrix = corr_matrix.copy()
    adj_matrix[pvals_corrected >= 0.05] = 0
    # 可选：增加相关性强度阈值
    # adj_matrix[np.abs(adj_matrix) < 0.3] = 0

    # 强制对称
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    print(adj_matrix.shape)
    # ================== 6. 保存结果 ==================
    np.save("{name}-adj_mat.npy".format(name=name), adj_matrix)

    print("✅ 相关性网络已保存为 adj_mat.npy")
    print(f"🔗 非零边数量: {np.count_nonzero(adj_matrix)}")

for name in abun_names:
    calc(name)