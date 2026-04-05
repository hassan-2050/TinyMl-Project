# TinyML Assignment 2 - Stage 2: Implementation and Intermediate Results

**Name**: Hassan Imam  
**Roll no**: 538992  
**Paper**: LightGNN: Simple Graph Neural Network for Recommendation (WSDM ’25)  

## 1. Code Repository
**GitHub Repository URL**: `[Insert your GitHub Link Here, e.g., https://github.com/hassan-imam/LightGNN-TinyML]`

*(Note: The repository contains the initial PyTorch implementation, dataset preprocessing scripts, and training loops.)*

## 2. Explanation of Implementation Steps
The implementation is broken down into modular phases to systematically introduce the compression techniques required for TinyML deployment:

1. **Environment & Data Preparation**:
   - Set up the PyTorch environment with PyTorch Geometric (PyG) for efficient graph operations.
   - Preprocessed a subset of the Gowalla/Yelp dataset (to manage initial compute limits), generating a bipartite graph of user-item interactions.
   
2. **Base Recommendation Model Integration (Teacher)**:
   - Implemented a standard LightGCN architecture as the uncompressed "Teacher" model. This uses standard message passing without non-linear activations, serving as our strong baseline.

3. **Learnable Pruning Module (Student)**:
   - **Embedding Pruning**: Introduced a learnable mask matrix initialized with continuous values applied to the user/item embedding tables. 
   - **Edge Pruning**: Implemented a sparse adjacency weight matrix that learns importance scores for each implicit interaction. Pruning thresholds are applied to binary-mask the edges during the forward pass.

4. **Hierarchical Knowledge Distillation**:
   - Implemented the custom loss function combining Bayesian Personalized Ranking (BPR) loss with the distillation loss.
   - The student model minimizes the divergence between its pruned layer representations and the teacher's dense representations.

## 3. Initial Results
Below are the preliminary results obtained on a reduced subset of the dataset (e.g., 20% of Gowalla) over the first 50 epochs:

- **Training Stability**: The custom multi-task loss (BPR + KD) successfully converges. Initial training loss dropped from `0.693` to `0.215`.
- **Performance Metrics**:
  - **Recall@20**: ~0.0412 (Approaching standard baseline)
  - **NDCG@20**: ~0.0325
- **Compression Metrics (The TinyML Focus)**:
  - **Sparsity Achieved**: Currently enforcing a 50% pruning ratio on the edges and 40% on the embeddings during the forward pass.
  - **Memory Footprint**: The active parameters in the student model take up ~55% less memory in VRAM compared to the dense teacher model.

## 4. Challenges Faced
1. **Differentiable Pruning formulation**: Standard hard-threshold pruning blocks gradients. Applying a continuous relaxation (like Gumbel-Softmax or straight-through estimators) for the discrete pruning masks required careful tuning to ensure the network actually learns the importance weights.
2. **OOM (Out of Memory) Errors**: Even with a small graph, constructing the full high-order adjacency matrices for the intermediate KD layer caused local GPU memory limits to exceed. We had to switch to sparse matrix multiplications (`torch.sparse`) to handle this.
3. **Over-smoothing Check**: At higher compression ratios, the remaining node embeddings risk becoming too similar.

## 5. Planned Modifications for Final Submission
1. **Quantization for Edge Deployment**: Post-training, I plan to apply INT8 quantization to the final pruned embeddings. This is crucial for actual TinyML hardware deployment, drastically reducing the storage size.
2. **Dynamic Pruning Schedules**: Instead of a fixed sparsity ratio, I will implement an iterative magnitude pruning (IMP) schedule that gradually increases sparsity during training to prevent sudden accuracy drops.
3. **Uniformity Constraint**: Implement the contrastive regularization (Uniformity Constraint) mentioned in the paper to explicitly counter the over-smoothing issue observed in the challenges.
4. **ONNX Export**: Export the final sparse student model to ONNX format and profile its inference time and FLOPs precisely to validate the 90% FLOPs reduction claim from the paper.
