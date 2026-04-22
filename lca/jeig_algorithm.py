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
        candidate_pool = self._generate_candidate_pool()
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


    def _gumbel_softmax_sample(self, p_pos: float, tau: float = 0.5) -> float:
        """Draws a soft label [-1, 1] using Gumbel-Softmax relaxation."""
        p_pos = np.clip(p_pos, 1e-10, 1.0 - 1e-10)
        p_neg = 1.0 - p_pos

        g_pos = -np.log(-np.log(np.random.uniform(1e-10, 1.0)))
        g_neg = -np.log(-np.log(np.random.uniform(1e-10, 1.0)))

        logit_pos = (np.log(p_pos) + g_pos) / tau
        logit_neg = (np.log(p_neg) + g_neg) / tau

        max_logit = max(logit_pos, logit_neg)
        exp_pos = np.exp(logit_pos - max_logit)
        exp_neg = np.exp(logit_neg - max_logit)

        sum_exp = exp_pos + exp_neg
        
        # Returns a continuous soft-label
        return (exp_pos / sum_exp * 1.0) + (exp_neg / sum_exp * -1.0)


    def _generate_candidate_pool(self) -> Dict[Tuple, float]:
        """
        Calculates entropy for ALL edges. Human-reviewed edges will naturally 
        have ~0 entropy and fall to the bottom. Returns the Top N most uncertain.
        """
        edge_entropies = {}
        
        for u, v in self.G.edges():
            edge_entropies[(u, v)] = self._calculate_entropy((u, v), self.Q)
            
        # Sort by highest entropy and take the top N (e.g., 50 or 500)
        sorted_edges = sorted(edge_entropies, key=edge_entropies.get, reverse=True)[:self.candidate_pool_size]
        
        return {edge: edge_entropies[edge] for edge in sorted_edges}


    def _run_monte_carlo_jeig(self, candidate_pool: Dict[Tuple, float]) -> List[Tuple]:
        logger.debug(f"Executing JEIG over Candidate Pool of {len(candidate_pool)}.")
        
        m = 5       # M subsets
        n = 50      # N configurations
        tau = 0.5   # Gumbel Temperature
        
        pool_edges = list(candidate_pool.keys())
        alpha_JEIG = {edge: 0.0 for edge in pool_edges}
        
        # Outer Loop: Select M subsets (size 1)
        for i in range(m):
            Di = random.sample(pool_edges, 1)
            
            Di_probs = []
            for edge in Di:
                u, v = edge
                idu, idv = self.node2idx[u], self.node2idx[v]
                p_pos = np.clip(np.sum(self.Q[idu] * self.Q[idv]), 1e-10, 1.0 - 1e-10)
                Di_probs.append((edge, p_pos))
            
            # Inner Loop: Sample N configurations
            for j in range(n):
                S_conditional = []
                
                # Sample configuration using Gumbel-Softmax
                for edge, p_pos in Di_probs:
                    soft_label = self._gumbel_softmax_sample(p_pos, tau=tau)
                    S_conditional.append((edge, soft_label))
                    
                # Rerun Mean Field with the SOFT configuration
                Q_cond = self._calculate_mean_field(self.M, S_conditional)
                
                # Accumulate the conditional entropy for the candidate pool
                for edge in pool_edges:
                    alpha_JEIG[edge] += (1.0 / n) * self._calculate_entropy(edge, Q_cond)
                    
        # Calculate Final Information Gain: I = H(e) - E[H(e|Di)]
        I_JEIG = {}
        for edge, base_entropy in candidate_pool.items():
            I_JEIG[edge] = base_entropy - (1.0 / m) * alpha_JEIG[edge]
            
        sorted_edges = sorted(I_JEIG, key=I_JEIG.get, reverse=True)
        
        if sorted_edges:
            top_score = I_JEIG[sorted_edges[0]]
            avg_score = np.mean(list(I_JEIG.values()))
            logger.info(f"JEIG Status | Top EIG: {top_score:.5f} | Avg EIG: {avg_score:.5f}")
            
            # Terminate if the graph is stable
            if top_score < 0.0001:
                logger.info("JEIG Converged: Max expected information gain is virtually zero.")
                self.phase = "FINISHED"
                return []
                
        return [(u, v, "human") for u, v in sorted_edges[:self.batch_size]]
    
    # def _get_unreviewed_edges(self) -> List[Tuple]:
    #     """Returns all edges in the graph that have not yet been reviewed by a human."""
    #     unreviewed_edges = []
    #     for u, v in self.G.edges():
    #         # Check your query history
    #         history = self.query_history.get((u, v), [])
    #         is_human_reviewed = any(ranker == 'human' for score, ranker in history)
            
    #         if not is_human_reviewed:
    #             unreviewed_edges.append((u, v))
                
    #     return unreviewed_edges
    
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
        