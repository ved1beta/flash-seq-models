import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class FlashAttention2:
    """
    Simplified Flash Attention 2 implementation in Python/PyTorch
    
    This implementation focuses on the core algorithmic ideas rather than
    low-level CUDA optimizations.
    """
    
    def __init__(self, block_size_q: int = 32, block_size_kv: int = 32):
        """
        Initialize Flash Attention 2
        
        Args:
            block_size_q: Block size for queries
            block_size_kv: Block size for keys/values
        """
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
    
    def _online_softmax_update(self, 
                              scores_block: torch.Tensor,
                              max_val: torch.Tensor,
                              sum_exp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Online softmax computation to maintain numerical stability
        
        Args:
            scores_block: Current attention scores block
            max_val: Running maximum values
            sum_exp: Running sum of exponentials
            
        Returns:
            Updated probabilities, max values, and sum of exponentials
        """
        # Compute new max values
        new_max = torch.maximum(max_val.unsqueeze(-1), scores_block.max(dim=-1, keepdim=True)[0])
        
        # Compute exponentials with numerical stability
        exp_scores = torch.exp(scores_block - new_max)
        exp_old = torch.exp(max_val.unsqueeze(-1) - new_max) * sum_exp.unsqueeze(-1)
        
        # Update running sums
        new_sum_exp = exp_old.squeeze(-1) + exp_scores.sum(dim=-1)
        
        # Compute probabilities for current block
        probs_block = exp_scores / new_sum_exp.unsqueeze(-1)
        
        return probs_block, new_max.squeeze(-1), new_sum_exp
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                scale: Optional[float] = None) -> torch.Tensor:
        """
        Flash Attention 2 forward pass
        
        Args:
            Q: Query tensor [batch_size, seq_len, head_dim]
            K: Key tensor [batch_size, seq_len, head_dim] 
            V: Value tensor [batch_size, seq_len, head_dim]
            mask: Optional attention mask
            scale: Attention scaling factor (defaults to 1/sqrt(head_dim))
            
        Returns:
            Output tensor [batch_size, seq_len, head_dim]
        """
        batch_size, seq_len_q, head_dim = Q.shape
        seq_len_kv = K.shape[1]
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # Initialize output and statistics
        O = torch.zeros_like(Q)  # Output accumulator
        l = torch.zeros(batch_size, seq_len_q, device=Q.device)  # Normalizer (sum of exp)
        m = torch.full((batch_size, seq_len_q), -torch.inf, device=Q.device)  # Max values
        
        # Number of blocks
        num_blocks_q = (seq_len_q + self.block_size_q - 1) // self.block_size_q
        num_blocks_kv = (seq_len_kv + self.block_size_kv - 1) // self.block_size_kv
        
        # Process query blocks
        for i in range(num_blocks_q):
            q_start = i * self.block_size_q
            q_end = min((i + 1) * self.block_size_q, seq_len_q)
            
            Q_block = Q[:, q_start:q_end, :]  # [batch, block_q, head_dim]
            O_block = torch.zeros_like(Q_block)
            l_block = torch.zeros(batch_size, q_end - q_start, device=Q.device)
            m_block = torch.full((batch_size, q_end - q_start), -torch.inf, device=Q.device)
            
            # Process key-value blocks
            for j in range(num_blocks_kv):
                kv_start = j * self.block_size_kv
                kv_end = min((j + 1) * self.block_size_kv, seq_len_kv)
                
                K_block = K[:, kv_start:kv_end, :]  # [batch, block_kv, head_dim]
                V_block = V[:, kv_start:kv_end, :]  # [batch, block_kv, head_dim]
                
                # Compute attention scores for this block
                scores_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale
                # [batch, block_q, block_kv]
                
                # Apply mask if provided
                if mask is not None:
                    mask_block = mask[q_start:q_end, kv_start:kv_end]
                    scores_block = scores_block.masked_fill(~mask_block, -torch.inf)
                
                # Online softmax update
                if j == 0:
                    # First block - initialize
                    m_new = scores_block.max(dim=-1)[0]  # [batch, block_q]
                    exp_scores = torch.exp(scores_block - m_new.unsqueeze(-1))
                    l_new = exp_scores.sum(dim=-1)  # [batch, block_q]
                    P_block = exp_scores / l_new.unsqueeze(-1)
                    
                    m_block = m_new
                    l_block = l_new
                    O_block = torch.matmul(P_block, V_block)
                else:
                    # Update with new block
                    m_new = torch.maximum(m_block, scores_block.max(dim=-1)[0])
                    
                    # Rescale previous output and statistics
                    exp_diff_old = torch.exp(m_block - m_new)
                    exp_diff_new = torch.exp(scores_block.max(dim=-1)[0] - m_new)
                    
                    exp_scores = torch.exp(scores_block - m_new.unsqueeze(-1))
                    l_new = exp_diff_old * l_block + exp_scores.sum(dim=-1)
                    
                    P_block = exp_scores / l_new.unsqueeze(-1)
                    
                    # Update output
                    O_block = (exp_diff_old / l_new).unsqueeze(-1) * O_block + \
                             torch.matmul(P_block, V_block)
                    
                    m_block = m_new
                    l_block = l_new
            
            # Store results for this query block
            O[:, q_start:q_end, :] = O_block
            l[:, q_start:q_end] = l_block
            m[:, q_start:q_end] = m_block
        
        return O

def test_flash_attention():
    """Test Flash Attention 2 implementation"""
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    head_dim = 64
    
    # Create test data
    Q = torch.randn(batch_size, seq_len, head_dim)
    K = torch.randn(batch_size, seq_len, head_dim)  
    V = torch.randn(batch_size, seq_len, head_dim)
    
    # Initialize Flash Attention
    fa2 = FlashAttention2(block_size_q=32, block_size_kv=32)
    
    # Compute Flash Attention output
    fa2_output = fa2.forward(Q, K, V)
    
    # Compare with standard attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    standard_output = torch.matmul(attn_weights, V)
    
    # Check if outputs are close
    max_diff = torch.max(torch.abs(fa2_output - standard_output))
    print(f"Maximum difference between FA2 and standard attention: {max_diff:.6f}")
    print(f"Outputs are close: {torch.allclose(fa2_output, standard_output, atol=1e-4)}")
    
    # Test with causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    fa2_causal = fa2.forward(Q, K, V, mask=causal_mask)
    
    # Standard causal attention for comparison
    scores_masked = scores.masked_fill(~causal_mask, -torch.inf)
    attn_weights_masked = F.softmax(scores_masked, dim=-1)
    standard_causal = torch.matmul(attn_weights_masked, V)
    
    max_diff_causal = torch.max(torch.abs(fa2_causal - standard_causal))
    print(f"Maximum difference with causal mask: {max_diff_causal:.6f}")
    print(f"Causal outputs are close: {torch.allclose(fa2_causal, standard_causal, atol=1e-4)}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing Flash Attention 2 Implementation")
    print("=" * 50)
    test_flash_attention()
    
    # Performance comparison example
    print("\nPerformance characteristics:")
    print("- Memory complexity: O(N) instead of O(N²)")
    print("- Maintains numerical stability through online softmax")  
    print("- Block-wise computation enables better cache utilization")
    print("- Exact attention computation (no approximation)")