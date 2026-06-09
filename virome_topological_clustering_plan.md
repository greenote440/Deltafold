# Training Specification: Multi-Rank Masked Topological Modeling (MTM)

## 1. Objective Function
The MTM objective is formulated as a self-supervised reconstruction task. The network is required to recover local structural information that has been intentionally masked, utilizing the latent hierarchical context provided by the higher-order ranks of the Protein Combinatorial Complex (PCC).

*   **Masking Strategy:** A subset of the Rank 0 (residue-level) features, specifically the structural alphabet tokens, are replaced with a `[MASK]` token.
*   **Reconstruction Target:** The model outputs a probability distribution over the original structural alphabet for each masked position.
*   **Loss Formulation:** The primary loss is the **Categorical Cross-Entropy** averaged over all masked nodes in the batch. By minimizing this loss, the network is forced to propagate structural information from the unmasked Rank 2 (Secondary Structure Elements) and Rank 3 (Global descriptors) to accurately predict the identity of the hidden Rank 0 tokens.

$$ \mathcal{L}_{\text{MTM}} = - \mathbb{E}_{i \in \mathcal{M}} \left[ \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) \right] $$

Where:
*   $\mathcal{M}$ represents the set of masked Rank 0 nodes.
*   $y_{i,c}$ is the ground-truth assignment for class $c$.
*   $\hat{y}_{i,c}$ is the predicted probability for class $c$ derived from the high-dimensional latent representation.

## 2. Training Loop Architecture
The training loop is structured to facilitate the iterative refinement of hierarchical embeddings without requiring pairwise alignments.

### Initialization
1.  **Topological Lifting:** Transform the raw 3D input coordinates into the PCC representation[cite: 3].
2.  **Embedding Initialization:** Initialize the TCPNet parameters and the projection head for the reconstruction task[cite: 3].

### Iterative Training Step
For each batch in the training set:
1.  **Stochastic Masking:** Perform a masking operation on the Rank 0 feature skeleton. Retain all spatial coordinates, but hide the biological/structural identity of a selected percentage of residues.
2.  **Hierarchical Forward Pass:** 
    *   Propagate signals through the TCPNet layers[cite: 3].
    *   Allow information to flow from the unmasked higher-order ranks (Rank 2 and Rank 3) down to the masked Rank 0 positions via the network's hierarchical message-passing pathways[cite: 3].
3.  **Local Feature Prediction:** Pass the updated Rank 0 latent representations through the reconstruction head to produce class probability distributions for the masked nodes[cite: 3].
4.  **Loss Computation:** Calculate the Categorical Cross-Entropy loss between the predictions and the ground-truth residue labels.
5.  **Parameter Update:** Perform backpropagation of the loss gradients to update the weights of the TCPNet, effectively optimizing the latent space to capture the global "structural fold" necessary for local feature inference[cite: 3].

## 3. Extraction for Downstream Clustering
Once the training objective is reached, the model has learned a structurally coherent embedding space. 

*   **Inference:** Pass unmasked, full-protein PCCs through the trained network.
*   **Representation Extraction:** Collect the global protein-level embedding vectors from the Rank 3 channel or via mean-pooling of the updated Rank 0 node embeddings[cite: 3].
*   **Clustering:** Use these fixed, dense vectors as inputs for standard clustering algorithms (e.g., HDBSCAN) within a high-performance vector index, such as FAISS, to achieve the required $\mathcal{O}(N \log N)$ scaling[cite: 3].# Training Specification: Multi-Rank Masked Topological Modeling (MTM)

## 1. Objective Function
The MTM objective is formulated as a self-supervised reconstruction task. The network is required to recover local structural information that has been intentionally masked, utilizing the latent hierarchical context provided by the higher-order ranks of the Protein Combinatorial Complex (PCC).

*   **Masking Strategy:** A subset of the Rank 0 (residue-level) features, specifically the structural alphabet tokens, are replaced with a `[MASK]` token.
*   **Reconstruction Target:** The model outputs a probability distribution over the original structural alphabet for each masked position.
*   **Loss Formulation:** The primary loss is the **Categorical Cross-Entropy** averaged over all masked nodes in the batch. By minimizing this loss, the network is forced to propagate structural information from the unmasked Rank 2 (Secondary Structure Elements) and Rank 3 (Global descriptors) to accurately predict the identity of the hidden Rank 0 tokens.

$$ \mathcal{L}_{\text{MTM}} = - \mathbb{E}_{i \in \mathcal{M}} \left[ \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) \right] $$

Where:
*   $\mathcal{M}$ represents the set of masked Rank 0 nodes.
*   $y_{i,c}$ is the ground-truth assignment for class $c$.
*   $\hat{y}_{i,c}$ is the predicted probability for class $c$ derived from the high-dimensional latent representation.

## 2. Training Loop Architecture
The training loop is structured to facilitate the iterative refinement of hierarchical embeddings without requiring pairwise alignments.

### Initialization
1.  **Topological Lifting:** Transform the raw 3D input coordinates into the PCC representation[cite: 3].
2.  **Embedding Initialization:** Initialize the TCPNet parameters and the projection head for the reconstruction task[cite: 3].

### Iterative Training Step
For each batch in the training set:
1.  **Stochastic Masking:** Perform a masking operation on the Rank 0 feature skeleton. Retain all spatial coordinates, but hide the biological/structural identity of a selected percentage of residues.
2.  **Hierarchical Forward Pass:** 
    *   Propagate signals through the TCPNet layers[cite: 3].
    *   Allow information to flow from the unmasked higher-order ranks (Rank 2 and Rank 3) down to the masked Rank 0 positions via the network's hierarchical message-passing pathways[cite: 3].
3.  **Local Feature Prediction:** Pass the updated Rank 0 latent representations through the reconstruction head to produce class probability distributions for the masked nodes[cite: 3].
4.  **Loss Computation:** Calculate the Categorical Cross-Entropy loss between the predictions and the ground-truth residue labels.
5.  **Parameter Update:** Perform backpropagation of the loss gradients to update the weights of the TCPNet, effectively optimizing the latent space to capture the global "structural fold" necessary for local feature inference[cite: 3].

## 3. Extraction for Downstream Clustering
Once the training objective is reached, the model has learned a structurally coherent embedding space. 

*   **Inference:** Pass unmasked, full-protein PCCs through the trained network.
*   **Representation Extraction:** Collect the global protein-level embedding vectors from the Rank 3 channel or via mean-pooling of the updated Rank 0 node embeddings[cite: 3].
*   **Clustering:** Use these fixed, dense vectors as inputs for standard clustering algorithms (e.g., HDBSCAN) within a high-performance vector index, such as FAISS, to achieve the required $\mathcal{O}(N \log N)$ scaling[cite: 3].