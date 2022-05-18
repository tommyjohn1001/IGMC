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


def sample_perm_matrix(n_nodes, device=None, dtype=None):
    indices = list(range(n_nodes))
    shuffled_indices = random.sample(indices, len(indices))
    perm_map = dict(zip(indices, shuffled_indices))

    perm_matrix = torch.zeros((n_nodes, n_nodes), device=device, dtype=dtype)
    for i, j in perm_map.items():
        perm_matrix[i][j] = 1

    return perm_matrix, perm_map


def sample_rot_matrix(dim, device, dtype):
    return torch.from_numpy(special_ortho_group.rvs(dim)).to(device=device, dtype=dtype)


def sample_translate_vec(dim, device, dtype):
    return torch.rand((1, dim), device=device, dtype=dtype)


def sample_transform_matrix(dim, device, dtype):
    perm_matrix = sample_perm_matrix(dim, device, dtype)[0]
    rot_matrix = sample_rot_matrix(dim, device, dtype)
    translt_vec = sample_translate_vec(dim, device, dtype)

    return perm_matrix @ rot_matrix + translt_vec


def create_trg_regu_matrix(
    trg_user_idx: Tensor,
    trg_item_idx: Tensor,
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
    trg_matrix[trg_user_idx] = trg_matrix[trg_item_idx] = beta
    trg_matrix[:, trg_user_idx] = trg_matrix[:, trg_item_idx] = beta
    trg_matrix[trg_user_idx, trg_item_idx] = trg_matrix[trg_item_idx, trg_user_idx] = alpha

    trg_matrix = trg_matrix + trg_matrix.triu().T

    return trg_matrix


def create_permuted_subgraph(
    data: Data,
    n_perm_subgraphs=4,
    pe_dim=5,
    metric: str = "L2",
    alpha: float = 0,
    beta: float = 2,
    device="cpu",
    dtype=torch.float,
    n_samples_equi=4,
) -> tuple:

    ## Get target user and item index
    trg_user_idx, trg_item_idx = (
        torch.where(data.x[:, 0] == 1)[0],
        torch.where(data.x[:, 1] == 1)[0],
    )

    ## Get matrix A and D
    D = pyg_utils.degree(data.edge_index[0])
    A = pyg_utils.to_dense_adj(data.edge_index).squeeze(0)
    n_nodes = len(D)

    permuted_graphs = []
    for _ in range(n_perm_subgraphs):
        perm_matrix, perm_map = sample_perm_matrix(n_nodes, device=device, dtype=dtype)

        new_A = perm_matrix @ A @ perm_matrix.T
        new_D = perm_matrix @ D
        pe_perm = get_rwpe_noscipy(new_A, new_D, pe_dim)

        new_user_idx, new_item_idx = perm_map[trg_user_idx.item()], perm_map[trg_item_idx.item()]
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

        ## Create ra
        pe_transforms, transforms = [], []
        for _ in range(n_samples_equi):
            transform_matrix = sample_transform_matrix(n_nodes, device=device, dtype=dtype)

            pe_transforms.append(transform_matrix @ pe_perm)
            transforms.append(transform_matrix)

        permuted_graphs.append((pe_perm, trg, pe_transforms, transforms))

    return permuted_graphs
