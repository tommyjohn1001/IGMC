import scipy.sparse as ssp
from all_packages import *
from scipy.stats import special_ortho_group
from torch_geometric.data import Data


def get_rwpe(A, D, pe_dim=5):

    # Geometric diffusion features with Random Walk
    A = ssp.csr_matrix(A.cpu().numpy())
    Dinv = ssp.diags(D.cpu().numpy().clip(1) ** -1.0)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pe_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())

    PE = torch.stack(PE, dim=-1)

    return PE


def get_rwpe_noscipy(A, D, pe_dim=5):

    # Geometric diffusion features with Random Walk
    Dinv = (D.clamp(1) ** -1.0).diag()
    RW = A @ Dinv
    M = RW

    # Iterate
    nb_pos_enc = pe_dim
    PE = [M.diagonal()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power @ M
        PE.append(M_power.diagonal())

    PE = torch.stack(PE, dim=-1)

    return PE


def sample_perm_matrix(batch: Tensor, device=None, dtype=None):
    permutation_matrix, permutation_map = [], dict()
    n = 0
    for b in batch.unique():
        n_nodes_each = torch.sum(batch == b).item()

        indices = list(range(n, n + n_nodes_each))
        shuffled_indices = random.sample(indices, len(indices))
        perm_map = dict(zip(indices, shuffled_indices))
        permutation_map = {**permutation_map, **perm_map}

        perm_matrix = torch.zeros((n_nodes_each, n_nodes_each), device=device, dtype=dtype)
        for i, j in perm_map.items():
            perm_matrix[i - n][j - n] = 1

        permutation_matrix.append(perm_matrix)

        n += n_nodes_each

    permutation_matrix = torch.block_diag(*permutation_matrix)

    return permutation_matrix, permutation_map


def sample_rot_matrix(dim, device, dtype):
    return torch.from_numpy(special_ortho_group.rvs(dim)).to(device=device, dtype=dtype)


def sample_translate_vec(dim, device, dtype):
    return torch.rand((1, dim), device=device, dtype=dtype)


def sample_transform_matrix(dim, batch, device, dtype):
    perm_matrix = sample_perm_matrix(batch, device, dtype)[0]
    rot_matrix = sample_rot_matrix(dim, device, dtype)
    translt_vec = sample_translate_vec(dim, device, dtype)

    return perm_matrix @ rot_matrix + translt_vec


def create_trg_regu_matrix(
    trg_user_indices: Tensor,
    trg_item_indices: Tensor,
    n_nodes: int,
    metric: str = "L2",
    alpha: float = 0,
    beta: float = 2,
    device=None,
    dtype=None,
):
    """Create target matrix for training regularization loss which is achieved by ContrastiveLoss"""

    if metric == "cosine":
        alpha, beta = 1, 0

    trg_matrix = torch.full((n_nodes, n_nodes), fill_value=alpha, device=device, dtype=dtype)
    for trg_user_idx, trg_item_idx in zip(trg_user_indices, trg_item_indices):
        trg_matrix[trg_user_idx] = beta
        trg_matrix[trg_user_idx, trg_item_idx] = alpha

    trg_matrix = trg_matrix + trg_matrix.triu().T

    return trg_matrix


def create_permuted_subgraph(
    data: Data,
    pe_dim=5,
    metric: str = "L2",
    alpha: float = 0,
    beta: float = 2,
    device=None,
    dtype=None,
) -> tuple:

    pe_perm, trg = [], []

    perm_matrix, perm_map = sample_perm_matrix(data.batch, device=device, dtype=dtype)

    ## Get target user and item index
    trg_user_idx, trg_item_idx = (
        torch.where(data.x[:, 0] == 1)[0],
        torch.where(data.x[:, 1] == 1)[0],
    )

    ## Get matrix A and D
    D = pyg_utils.degree(data.edge_index[0])
    A = pyg_utils.to_dense_adj(data.edge_index).squeeze(0)

    new_A = perm_matrix @ A @ perm_matrix.T
    new_D = perm_matrix @ D
    pe_perm = get_rwpe_noscipy(new_A, new_D, pe_dim)

    new_user_idx = [perm_map[x.item()] for x in trg_user_idx]
    new_item_idx = [perm_map[x.item()] for x in trg_item_idx]
    trg = create_trg_regu_matrix(
        new_user_idx,
        new_item_idx,
        n_nodes=len(D),
        metric=metric,
        alpha=alpha,
        beta=beta,
        device=device,
        dtype=dtype,
    )

    return pe_perm, trg
