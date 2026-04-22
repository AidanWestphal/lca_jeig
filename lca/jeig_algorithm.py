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

        # 2. Re-calculate Base Clustering (PCCs) & Build Similarity Matrix
        clusters = self._calculate_pcc_clusters()
        self._build_similarity_matrix()
        
        self.M = self._build_M_matrix(clusters)

        # 3. Calculate Mean Field Approximation (Global Q)
        self.Q = self._calculate_mean_field(self.M)
        
        # 4. Generate Candidate Pool (Uncertain edges -- haven't been viewed by a human)
        candidate_pool = self._get_candidate_pool()
        if not candidate_pool:
            logger.info("No candidates available. Marking as finished.")
            self.phase = "FINISHED"
            return []

        # 5. Run Monte Carlo JEIG on candidates to find best batch
        best_batch = self._run_monte_carlo_jeig(candidate_pool)
        
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
        clusters = self._calculate_pcc_clusters()
        node2cid = {}
        cluster_dict = {}
        
        for cid, cluster_nodes in enumerate(clusters):
            cluster_dict[cid] = list(cluster_nodes)
            for node in cluster_nodes:
                node2cid[node] = cid
                
        return cluster_dict, node2cid, self.G

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
            
            # Seen by human, add human review to overall count
            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                self.num_human_reviews += 1
                # No high scores or anything, just clean. Human can give uncertain 0.5 and be fine
                math_score = score
                ranker = "human"
            else:
                math_score = score
                ranker = verifier_name
            
            # Add query history
            if (u, v) not in self.query_history:
                self.query_history[(u, v)] = []
                
            self.query_history[(u, v)].append((math_score, ranker))
            
            # Average query history for new score
            avg_score = self._mean_query_history(u, v)
            
            # Update in graph
            if self.G.has_edge(u, v):
                self.G.edges[u, v]["score"] = avg_score
                self.G.edges[u, v]["ranker"] = ranker
            else:
                self.G.add_edge(u, v, score=avg_score, ranker=ranker)

    def _calculate_pcc_clusters(self):
        # Read current graph to generate clustering
        positive_edges = [
            (u, v) for u, v, data in self.G.edges(data=True) 
            if data['score'] >= self.threshold
        ]
        pos_G = self.G.edge_subgraph(positive_edges)
        
        clusters = [list(c) for c in nx.connected_components(pos_G)]
        
        clustered_nodes = set().union(*clusters) if clusters else set()
        singletons = [[n] for n in self.G.nodes() if n not in clustered_nodes]
        
        clusters.extend(singletons)
        return clusters

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
            # Zero-centered similarity [0,1] -> [-1, 1]
            val = 2 * (data['score'] - self.threshold)
            self.S[i, j] = val
            self.S[j, i] = val
    
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

    def _get_candidate_pool(self) -> List[Tuple]:
        # For simplicity we only care about non-viewed edges
        pool = []
        for u, v, data in self.G.edges(data=True):
            if data.get('ranker') != 'human':
                pool.append((u, v))
                
        return pool[:self.candidate_pool_size]
    
    def _calculate_entropy(self, edge, Q):
        # Calculate entropy for an edge conditional on Q probs
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

    def _run_monte_carlo_jeig(self, candidate_pool: List[Tuple]) -> List[Tuple]:
        logger.debug("Executing Monte Carlo Info Gain")
        alpha_JEIG = {}

        # 1. Outer Loop: Sample subsets of edges
        sample_size = max(1, int(len(candidate_pool) * self.pair_sample_fraction))
        
        for i in range(self.pair_sample_count):
            Di = random.sample(candidate_pool, sample_size)
            
            # 2. Inner Loop: Monte Carlo over cluster assignments
            for j in range(self.cluster_sample_count):
                # Sample random edge assignments specifically for the subset Di
                edges = np.random.choice([-1.0, 1.0], size=len(Di))
                S_conditional = list(zip(Di, edges))
                
                # Get conditional probabilities
                Q_di = self._calculate_mean_field(self.M, S_conditional)
                
                # Update JEIG scores
                for edge in Di:
                    if edge not in alpha_JEIG:
                        alpha_JEIG[edge] = 0.0
                    alpha_JEIG[edge] += (1 / self.cluster_sample_count) * self._calculate_entropy(edge, Q_di)
        
        # 3. Final calculations: H(edge) - Expected[H(edge | Di)]
        for edge, expected_conditional_entropy in alpha_JEIG.items():
            base_entropy = self._calculate_entropy(edge, self.Q)
            alpha_JEIG[edge] = base_entropy - (1 / self.pair_sample_count) * expected_conditional_entropy
        
        sorted_edges = sorted(alpha_JEIG, key=alpha_JEIG.get, reverse=True)
        return [(u, v, "human") for u, v in sorted_edges[:self.batch_size]]
    

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
        num_clusters = len(self._calculate_pcc_clusters()) if self.G.nodes() else 0
        stats_logger.info(f"Final Clusters (PCCs): {num_clusters}")
        stats_logger.info("=" * 60)
        