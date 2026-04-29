import itertools
import networkx as nx
import numpy as np
import scipy.sparse as sp
import logging
from typing import Dict, List, Tuple, Set, Optional
import random

logger = logging.getLogger('lca')

class LCAv3JEIGAlgorithm:
    """
    LCA v3 Joint Entropy Information Gain (JEIG) Algorithm.
    
    Implements Correlation Clustering active learning via Mean Field 
    and Monte Carlo approximations, assuming a noisy oracle.
    """
    def __init__(self, config: Dict, classifier_manager, cluster_validator=None):
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator
        
        # Graph & State
        self.G = nx.Graph()
        self.query_history = {}  # {(u, v): [(score, ranker), ...]}
        
        # JEIG Core Parameters
        jeig_params = self.config.get('jeig', {})
        self.batch_size = jeig_params.get('batch_size', 10)
        self.candidate_pool_size = jeig_params.get('candidate_pool_size', 50)
        self.beta = jeig_params.get('beta', 1.0)
        self.threshold = jeig_params.get('threshold', 0.5)
        self.human_weight = jeig_params.get('human_weight', 2)
        self.pair_sample_count = jeig_params.get('pair_sample_count', 5)
        self.pair_sample_fraction = jeig_params.get('pair_sample_fraction', 0.02)
        self.cluster_sample_count = jeig_params.get('cluster_sample_count', 50)
        
        # Termination Parameters
        self.max_human_reviews = jeig_params.get('max_human_reviews', 5000)
        self.num_human_reviews = 0
        self.phase = "ACTIVE" # ACTIVE -> FINISHED
        
        # Global Mathematical State Variables
        self.S = None            
        self.Q = None            
        self.M = None
        self.node2idx = {}       
        self.idx2node = {}       
        
        logger.info("LCA v3 JEIG Algorithm initialized")


    def step(self, new_edges: List[Tuple]) -> List[Tuple[int, int, float]]:
        if self.is_finished():
            return []

        # 1. Process and add edges (Initial & Human)
        self._process_and_add_edges(new_edges)
        
        if len(self.G.nodes()) < 2:
            return []

        # 2. Re-calculate Base Clustering (Local Search) & Build Similarity Matrix
        clusters = self._calculate_local_search_clusters()
        self._build_similarity_matrix()
        
        self.M = self._build_M_matrix(clusters)

        # 3. Calculate Mean Field Approximation (Global Q)
        self.Q = self._calculate_mean_field(self.M)

        # 5. Run Monte Carlo JEIG on candidates to find best batch
        best_batch = self._run_monte_carlo_jeig()
        
        return best_batch


    def is_finished(self) -> bool:
        if self.phase == "FINISHED":
            return True
        if self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"JEIG reached max human reviews ({self.max_human_reviews})")
            self.phase = "FINISHED"
            return True
        return False


    def get_clustering(self) -> Tuple[Dict, Dict, nx.Graph]:
        # Recalculate clusters for the output signature
        clusters = self._calculate_local_search_clusters()
        node2cid = {}
        cluster_dict = {}
        
        for cid, cluster_nodes in enumerate(clusters):
            cluster_dict[cid] = list(cluster_nodes)
            for node in cluster_nodes:
                node2cid[node] = cid
                
        return cluster_dict, node2cid, self.G


    def _log_incremental_metrics(self, max_jeig_score=None):
        """Logs Hungarian metrics and JEIG scores during iterations for graph parsing."""
        if not self.cluster_validator:
            return

        # Get current clustering state
        clustering, node2cid, _ = self.get_clustering()

        # =========================================================
        # FIX: Convert clustering lists to sets! 
        # LCA's cluster_tools.py expects sets to use the .add() method.
        # =========================================================
        clustering_as_sets = {cid: set(nodes) for cid, nodes in clustering.items()}

        # Extract the ground truth data from the validator
        true_clustering = self.cluster_validator.gt_clustering
        true_node2cid = self.cluster_validator.gt_node2cid

        # Calculate and log Hungarian F1, Recall, and Precision
        self.cluster_validator.incremental_stats(
            self.num_human_reviews,
            clustering_as_sets,  # Pass the sets here!
            node2cid,
            true_clustering,
            true_node2cid
        )

        # Log the JEIG score trend explicitly for easy parsing
        if max_jeig_score is not None:
            logger.info(f"JEIG_Trend num_human={self.num_human_reviews}, max_jeig_score={max_jeig_score:.8f}")


    def _mean_query_history(self, u, v):
        # Weighted mean over query history for an edge u - v
        sum_score = 0
        count_score = 0
        for score, ranker in self.query_history[(u, v)]:
            # Weight human decisions heavier
            if ranker == 'human':
                sum_score += score * self.human_weight
                count_score += self.human_weight
            else:
                sum_score += score
                count_score += 1

        return sum_score / count_score if count_score > 0 else 0


    def _process_and_add_edges(self, new_edges: List[Tuple]):
        # Over all new edges...
        for edge in new_edges:
            n0, n1, score, verifier_name = edge[:4]
            u, v = min(n0, n1), max(n0, n1)
            
            # Ensure nodes are registered in the graph
            if not self.G.has_node(u): self.G.add_node(u)
            if not self.G.has_node(v): self.G.add_node(v)
            

            # Seen by human, add human review to overall count
            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                self.num_human_reviews += 1
                ranker = "human"
                # manual assignment
                confidence = 1 if score > 0.5 else -1
                # print(f"HUMAN EDGE {confidence}")
            else:
                classified = self.classifier_manager.classify_edge(n0, n1, verifier_name)
                _, _, _, confidence, label, ranker = classified

                # If negative edge, flip confidence to negative scale
                if label == "negative":
                    confidence *= -1

                # print(f"EDGE {confidence} {label} {verifier_name} {ranker}")

            # Add query history
            if (u, v) not in self.query_history:
                self.query_history[(u, v)] = []
                
            self.query_history[(u, v)].append((confidence, ranker))
            
            # Average query history for new score
            avg_score = self._mean_query_history(u, v)
            
            # Update in graph
            if self.G.has_edge(u, v):
                self.G.edges[u, v]["score"] = avg_score
                self.G.edges[u, v]["ranker"] = ranker
            else:
                self.G.add_edge(u, v, score=avg_score, ranker=ranker)


    def _calculate_local_search_clusters(self):
        # Local Search Algorithm for Correlation Clustering (Alg. 2)
        nodes = list(self.G.nodes())
        if not nodes:
            return []
            
        # 1: Randomly assign each object to a cluster
        node2cid = {node: i for i, node in enumerate(nodes)}
        clusters = {i: {node} for i, node in enumerate(nodes)}
        
        changed = True
        max_iters = 100
        iters = 0
        
        # 2: while not converged do
        while changed and iters < max_iters:
            changed = False
            # 3: Select object i randomly
            random.shuffle(nodes) 
            
            for u in nodes:
                current_cid = node2cid[u]
                
                # 4: Assign object i to the cluster that maximally increases the objective
                cluster_sims = {}
                for v in self.G.neighbors(u):
                    v_cid = node2cid[v]
                    cluster_sims[v_cid] = cluster_sims.get(v_cid, 0) + self.G.edges[u, v]['score']
                
                best_cid = None
                best_sim = 0.0 
                
                for cid, sim in cluster_sims.items():
                    if sim > best_sim:
                        best_sim = sim
                        best_cid = cid
                        
                # If no existing cluster yields an increase, form a new cluster
                if best_cid is None:
                    best_cid = max(clusters.keys(), default=0) + 1
                    clusters[best_cid] = set()
                    
                if best_cid != current_cid:
                    clusters[current_cid].remove(u)
                    if not clusters[current_cid]:
                        del clusters[current_cid]
                    
                    clusters[best_cid].add(u)
                    node2cid[u] = best_cid
                    changed = True
                    
            iters += 1
            
        return [list(c) for c in clusters.values() if c]


    def _build_similarity_matrix(self):
        # Build similarity matrix for the current graph
        nodes = list(self.G.nodes())
        n_nodes = len(nodes)
        
        self.node2idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx2node = {idx: node for idx, node in enumerate(nodes)}
        
        # Init to zeros -- sparse except for edges present
        self.S = np.zeros((n_nodes, n_nodes))
        
        for u, v, data in self.G.edges(data=True):
            i, j = self.node2idx[u], self.node2idx[v]

            self.S[i, j] = data['score']
            self.S[j, i] = data['score']
    

    def _build_M_matrix(self, clustering):
        # Build affinity matrix (scores/cost of cluster assignment)
        n_nodes = len(self.node2idx)
        n_clusters = len(clustering)
        
        M = np.zeros((n_nodes, n_clusters))
        
        for k, cluster_nodes in enumerate(clustering):
            cluster_indices = [self.node2idx[n] for n in cluster_nodes]
            if cluster_indices:
                # Paper uses negative energy system
                M[:, k] = -np.sum(self.S[:, cluster_indices], axis=1)
        
        return M


    def _calculate_mean_field(self, M_init, S_conditional=[]):
        """Calculates Q: The probability distribution of nodes over clusters."""
        # Prevent overwriting M
        M = M_init.copy()
        Q = np.zeros(M.shape)
        S = self.S.copy()

        # Apply conditional edge flips
        for (u, v), score in S_conditional:
            idu, idv = self.node2idx[u], self.node2idx[v]
            # Symmetry
            S[idu, idv] = score
            S[idv, idu] = score

        # While Q is not converged... (and safeguard)
        max_iters = 50
        for _ in range(max_iters):
            # Softmaxxing
            exp_Q = np.exp(-self.beta * M)
            exp_Q = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
            
            # Update Energy field M based on new probabilities
            M = -S @ exp_Q
            
            # Check convergence
            if np.linalg.norm(exp_Q - Q) < 0.001:
                return exp_Q
                
            Q = exp_Q
            
        return Q


    def _calc_entropies_fast(self, U_idx: np.ndarray, V_idx: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Vectorized calculation of entropy for a batch of edges."""
        p_pos = np.sum(Q[U_idx] * Q[V_idx], axis=1)
        
        # Epsilon safeguard to prevent np.log2(0) == -inf crashes
        eps = 1e-10
        p_pos = np.clip(p_pos, eps, 1.0 - eps)
        p_neg = 1.0 - p_pos
        
        return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)


    def _run_monte_carlo_jeig(self) -> List[Tuple]:
        # Filter down to known edges only per Top-K strategy
        pool_edges = list(self.G.edges)
        if not pool_edges:
            return []
            
        num_edges = len(pool_edges)
        logger.debug(f"Executing JEIG over {num_edges} edges.")
        
        m = self.pair_sample_count
        n = self.cluster_sample_count
        sample_size = self.batch_size if isinstance(self.batch_size, int) else max(1, int(self.batch_size * num_edges))
        
        # Pre-extract indices to vectorize inner loops
        U_idx = np.array([self.node2idx[u] for u, v in pool_edges])
        V_idx = np.array([self.node2idx[v] for u, v in pool_edges])
        
        # Initialize accumulation array (much faster than dict)
        alpha_JEIG_arr = np.zeros(num_edges)
        
        # Vectorized base entropy calculation
        base_entropies_arr = self._calc_entropies_fast(U_idx, V_idx, self.Q)
        base_entropies = dict(zip(pool_edges, base_entropies_arr))
        
        # Outer Loop: Select M subsets
        for i in range(m):
            # Select top-|Di| pairs using Gumbel + Log Entropy
            noisy_entropies = {
                edge: np.log2(max(ent, 1e-10)) + np.random.gumbel() 
                for edge, ent in base_entropies.items()
            }
            sorted_by_noisy_ent = sorted(noisy_entropies, key=noisy_entropies.get, reverse=True)
            Di = sorted_by_noisy_ent[:sample_size]
            
            # Precalculate probability values
            Di_probs = []
            for edge in Di:
                u, v = edge
                idu, idv = self.node2idx[u], self.node2idx[v]
                p_pos = np.clip(np.sum(self.Q[idu] * self.Q[idv]), 1e-10, 1.0 - 1e-10)
                Di_probs.append((edge, p_pos))
            
            # Inner Loop: Sample N configurations
            for j in range(n):
                S_conditional = []
                
                # Sample configuration
                for edge, p_pos in Di_probs:
                    # Select off Di probs
                    edge_label = np.random.choice([1, -1], p=[p_pos, 1 - p_pos])
                    S_conditional.append((edge, edge_label))
                
                # Rerun Mean Field with the sampled labels
                Q_cond = self._calculate_mean_field(self.M, S_conditional)
                
                # Fast Vectorized Accumulation
                alpha_JEIG_arr += (1.0 / n) * self._calc_entropies_fast(U_idx, V_idx, Q_cond)
                    
        # Calculate Final Information Gain: I = H(e) - E[H(e|Di)] w/ gumbel noise and log
        I_JEIG = {}
        for idx, edge in enumerate(pool_edges):
            raw_ig = base_entropies_arr[idx] - (1.0 / m) * alpha_JEIG_arr[idx]
            I_JEIG[edge] = np.log2(max(raw_ig, 1e-10)) + np.random.gumbel(loc=0.0, scale=1.0)
            
        sorted_edges = sorted(I_JEIG, key=I_JEIG.get, reverse=True)
        
        if sorted_edges:
            top_score = I_JEIG[sorted_edges[0]]
            avg_score = np.mean(list(I_JEIG.values()))
            logger.info(f"JEIG Status | Top EIG: {top_score:.5f} | Avg EIG: {avg_score:.5f}")

            # =========================================================
            # LOG METRICS BEFORE ASKING HUMAN OR TERMINATING
            # This handles F1, Recall, Precision, and the JEIG trend
            # =========================================================
            self._log_incremental_metrics(max_jeig_score=top_score)
            
            # Terminate if the graph is stable (check raw IG, not noisy)
            best_idx = pool_edges.index(sorted_edges[0])
            top_raw_ig = base_entropies_arr[best_idx] - (1.0 / m) * alpha_JEIG_arr[best_idx]
            # if top_raw_ig < 0.00001:
            #     logger.info("JEIG Converged: Max expected information gain is virtually zero.")
            #     self.phase = "FINISHED"
            #     return []
        
        # Gumbel sampling of top batch size
        return [(u, v, "human") for u, v in sorted_edges[:self.batch_size]]


    def _calculate_entropy(self, edge, Q):
        """Kept for backward compatibility if needed, though replaced by _calc_entropies_fast in hot loops."""
        u, v = edge
        idu, idv = self.node2idx[u], self.node2idx[v]
        
        Qu = Q[idu]
        Qv = Q[idv]
        
        p_pos = np.sum(Qu * Qv)
        p_neg = 1.0 - p_pos
        
        # Epsilon safeguard to prevent np.log2(0) == -inf crashes
        eps = 1e-10
        p_pos = np.clip(p_pos, eps, 1.0 - eps)
        p_neg = np.clip(p_neg, eps, 1.0 - eps)
        
        return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)
    

    def show_stats(self):
        """Logs algorithm statistics at the end of the run for run.py."""
        stats_logger = logging.getLogger('lca')
        stats_logger.info("=" * 60)
        stats_logger.info("LCA v3 JEIG Algorithm Statistics")
        stats_logger.info("=" * 60)
        stats_logger.info(f"Phase: {self.phase}")
        stats_logger.info(f"Human reviews consumed: {self.num_human_reviews}")
        stats_logger.info(f"Final Nodes: {len(self.G.nodes())}")
        stats_logger.info(f"Final Edges: {len(self.G.edges())}")
        
        # Calculate clusters if not None
        num_clusters = len(self._calculate_local_search_clusters()) if self.G.nodes() else 0
        stats_logger.info(f"Final Clusters (Local Search): {num_clusters}")
        stats_logger.info("=" * 60)