import multiprocessing

import numpy as np
import scipy.sparse as ssp
import torch.multiprocessing
from all_packages import *
from preprocessing import load_data_monti
from torch_geometric.data import Data

torch.multiprocessing.set_sharing_strategy("file_system")


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def subgraph_extraction_labeling(
    ind,
    Arow,
    Acol,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
    y=1,
):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h + 1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio * len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0

    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    node_index = u_nodes + v_nodes
    max_node_label = 2 * h + 1
    y = class_values[y]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None

    # only output node features for the target user and item
    if u_features is not None and v_features is not None:
        node_features = [u_features[0], v_features[0]]

    return u, v, r, node_labels, max_node_label, y, node_features, node_index


def get_non_edges(example: Data, k: int = 50) -> list:
    user_node_idx = torch.where((example.x[:, 1] == 1) | (example.x[:, 3] == 1))
    item_node_idx = torch.where((example.x[:, 2] == 1) | (example.x[:, 4] == 1))
    users = set(user_node_idx[0].tolist())
    items = set(item_node_idx[0].tolist())

    edge_index = example.edge_index.permute((1, 0))
    edges = set(tuple(x) for x in edge_index.tolist())
    possible_edges = set(itertools.product(users, items))

    non_edges = possible_edges.difference(edges)
    if non_edges == []:
        return []

    ## Take k edges only
    k = min(k, len(non_edges))
    non_edges = random.sample(non_edges, k=k)

    return non_edges


def get_rwpe(A, D, pos_enc_dim=5):

    # Geometric diffusion features with Random Walk
    A = ssp.csr_matrix(A.numpy())
    Dinv = ssp.diags(D.numpy().clip(1) ** -1.0)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())

    PE = torch.stack(PE, dim=-1)

    return PE


def construct_pyg_graph(
    u, v, r, node_labels, max_node_label, y, node_features, node_index, pos_enc_dim
):
    ## This condition to ensure graphs without neighbors are discarded
    if len(u) == 0:
        return None

    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    node_index = torch.LongTensor(node_index)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_type=edge_type, y=y)

    return data


def create_permute_matrix(size):
    indices = list(range(size))
    random.shuffle(indices)

    permutation_map = {old: new for old, new in zip(range(size), indices)}

    permutation_matrix = torch.zeros((size, size))
    for i, j in enumerate(indices):
        permutation_matrix[i][j] = 1

    return permutation_matrix, permutation_map


def create_trg_regu_matrix(trg_user_idx, trg_item_idx, n_nodes: int):
    """Create target matrix for training regularization loss which is achieved by ContrastiveLoss"""

    trg_matrix = torch.ones((n_nodes, n_nodes))
    S = set([trg_user_idx, trg_item_idx])

    for i in range(n_nodes):
        for j in range(n_nodes):
            condition1 = i not in S and j in S
            condition2 = i in S and j not in S
            if condition1 or condition2:
                trg_matrix[i, j] = 0

    return trg_matrix


def create_permuted_graphs(data: Data, n=10, pos_enc_dim=5) -> list:
    if len(data.edge_index[0]) == 0:
        x_perms = torch.zeros((n, data.x.size(0) + pos_enc_dim))
        targets_perms = torch.zeros((n, 2))

        return x_perms, targets_perms

    ## Get matrix A and D
    D = pyg_utils.degree(data.edge_index[0])
    A = pyg_utils.to_dense_adj(data.edge_index).squeeze(0)

    ## Get target user and item index
    trg_user_idx, trg_item_idx = (
        torch.where(data.x[:, 0] == 1)[0].item(),
        torch.where(data.x[:, 1] == 1)[0].item(),
    )

    permuted_graphs = []
    for _ in range(n):
        perm_matrix, perm_map = create_permute_matrix(len(D))

        new_A = perm_matrix @ A @ perm_matrix.T
        new_D = perm_matrix @ D
        x = get_rwpe(new_A, new_D, pos_enc_dim)

        new_user_idx, new_item_idx = perm_map[trg_user_idx], perm_map[trg_item_idx]
        trg = create_trg_regu_matrix(new_user_idx, new_item_idx, len(D))

        permuted_graphs.append((x.numpy(), trg.numpy()))

    return permuted_graphs


class ParallelHelper:
    def __init__(
        self,
        f_task: object,
        data: list,
        data_allocation: object,
        num_workers,
        desc=None,
        show_bar=False,
        n_data=None,
        *args,
    ):
        self.n_data = len(data) if n_data is None else n_data
        self.show_bar = show_bar

        self.queue = multiprocessing.Queue()
        if self.show_bar:
            self.pbar = tqdm(total=self.n_data, desc=desc)

        self.jobs = list()
        for ith in range(num_workers):
            lo_bound = ith * self.n_data // num_workers
            hi_bound = (
                (ith + 1) * self.n_data // num_workers if ith < (num_workers - 1) else self.n_data
            )

            p = multiprocessing.Process(
                target=f_task,
                args=(data_allocation(data, lo_bound, hi_bound), self.queue, *args),
            )
            self.jobs.append(p)

    def launch(self) -> list:
        """
        Launch parallel process
        Returns: a list after running parallel task
        """
        dataset = []

        for job in self.jobs:
            job.start()

        cnt = 0
        while cnt < self.n_data:
            while not self.queue.empty():
                dataset.append(self.queue.get())
                cnt += 1

                if self.show_bar:
                    self.pbar.update()

        if self.show_bar:
            self.pbar.close()

        for job in self.jobs:
            job.terminate()

        for job in self.jobs:
            job.join()

        return dataset


def create_data(
    args,
    train_ratio=0.8,
    parallel=True,
    num_workers=8,
):
    g_list = []
    (
        u_features,
        v_features,
        adj_train,
        train_labels,
        train_u_indices,
        train_v_indices,
        val_labels,
        val_u_indices,
        val_v_indices,
        test_labels,
        test_u_indices,
        test_v_indices,
        class_values,
        n_nodes,
    ) = load_data_monti(args.dataset, testing=True)

    Arow = SparseRowIndexer(adj_train)
    Acol = SparseColIndexer(adj_train.tocsc())

    ## Create list of graphs
    def core(u_indices, v_indices, labels) -> list:
        g_lists_output = []
        with tqdm(total=len(labels)) as pbar:
            for i, j, g_label in zip(u_indices, v_indices, labels):
                tmp = subgraph_extraction_labeling(
                    (i, j),
                    Arow,
                    Acol,
                    args.hop,
                    args.sample_ratio,
                    args.max_nodes_per_hop,
                    u_features,
                    v_features,
                    class_values,
                    g_label,
                )
                data = construct_pyg_graph(*tmp, args.pe_dim)
                if data is not None:
                    g_lists_output.append(data)

                pbar.update()

        return g_lists_output

    g_list.extend(core(train_u_indices, train_v_indices, train_labels))
    g_list.extend(core(test_u_indices, test_v_indices, test_labels))

    g_list = g_list[:12000]

    logger.info(f"No. graphs created: {len(g_list)}")

    ## For each subgraph, create permutated versions
    permuted_graphs = []
    if not parallel:
        for g in tqdm(g_list, desc="Create permuted graphs"):
            permuted_graphs += create_permuted_graphs(g, args.n_perm_graphs, args.pe_dim)
    else:

        def f_create(data_list, queue, n, pos_enc_dim):
            for data in data_list:
                a = create_permuted_graphs(data, n, pos_enc_dim)
                for x in a:
                    queue.put(x)

        permuted_graphs = ParallelHelper(
            f_create,
            g_list,
            lambda dat, l, h: dat[l:h],
            num_workers,
            None,
            True,
            len(g_list) * args.n_perm_graphs,  # Nếu đúng thì chỗ này phải nhân với n_perm_graphs
            args.n_perm_graphs,
            args.pe_dim,
        ).launch()

    logger.info(f"No. permuted graphs created: {len(permuted_graphs)}")

    ## Split into train/val split
    random.shuffle(permuted_graphs)
    n_train_samples = int(train_ratio * len(permuted_graphs))
    data_train, data_val = permuted_graphs[:n_train_samples], permuted_graphs[n_train_samples:]

    return data_train, data_val
