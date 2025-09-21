import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = config.beta  # KL penalty
        self.group_size = config.group_size
        
    def compute_group_rewards(self, responses, rewards):
        """
        Compute relative rewards within groups
        
        Args:
            responses: List of response strings
            rewards: List of scalar rewards for each response
        """
        group_rewards = []
        
        for i in range(0, len(responses), self.group_size):
            group = rewards[i:i+self.group_size]
            relative_rewards = []
            
            for j in range(len(group)):
                # Relative reward = average difference with others
                others = [group[k] for k in range(len(group)) if k != j]
                rel_reward = np.mean([group[j] - other for other in others])
                relative_rewards.append(rel_reward)
            
            group_rewards.extend(relative_rewards)
            
        return torch.tensor(group_rewards)
    
    def compute_policy_loss(self, query_tensors, response_tensors, 
                           rewards, old_log_probs):
        """
        Compute GRPO policy loss
        """
        batch_size = len(query_tensors)
        
        # Get current policy log probabilities
        all_log_probs = []
        all_ref_log_probs = []
        
        for i in range(batch_size):
            # Concatenate query and response
            input_ids = torch.cat([query_tensors[i], response_tensors[i]])
            
            # Forward pass through current model
            with torch.no_grad():
                outputs = self.model(input_ids.unsqueeze(0))
                logits = outputs.logits[0, len(query_tensors[i])-1:-1]
                
                # Get log probabilities for response tokens
                response_log_probs = F.log_softmax(logits, dim=-1)
                response_tokens = response_tensors[i]
                
                log_prob = response_log_probs.gather(
                    1, response_tokens.unsqueeze(1)
                ).squeeze().sum()
                all_log_probs.append(log_prob)
            
            # Forward pass through reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids.unsqueeze(0))
                ref_logits = ref_outputs.logits[0, len(query_tensors[i])-1:-1]
                
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_log_prob = ref_log_probs.gather(
                    1, response_tokens.unsqueeze(1)
                ).squeeze().sum()
                all_ref_log_probs.append(ref_log_prob)
        
        current_log_probs = torch.stack(all_log_probs)
        ref_log_probs = torch.stack(all_ref_log_probs)
        
        # Compute log ratio
        log_ratios = current_log_probs - ref_log_probs
        
        # GRPO loss
        policy_loss = -(rewards * log_ratios).mean()
        
        # KL penalty
        kl_penalty = self.beta * (current_log_probs - ref_log_probs).mean()
        
        total_loss = policy_loss + kl_penalty
        
        return total_loss, policy_loss, kl_penalty
    
    def train_step(self, batch):
        """
        Single training step
        """
        query_tensors = batch['query_tensors']
        response_tensors = batch['response_tensors']
        scores = batch['scores']  # Human preference scores
        
        # Compute group-based relative rewards
        rewards = self.compute_group_rewards(
            batch['responses'], scores
        )
        
        # Compute loss
        loss, policy_loss, kl_penalty = self.compute_policy_loss(
            query_tensors, response_tensors, rewards, None
        )
        
        return {
            'loss': loss,
            'policy_loss': policy_loss,
            'kl_penalty': kl_penalty,
            'mean_reward': rewards.mean()
        }
class RewardModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = torch.nn.Linear(
            base_model.config.hidden_size, 1
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # Use last token representation
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

def compute_rewards(reward_model, queries, responses):
    """Compute rewards for query-response pairs"""
    rewards = []
    
    for query, response in zip(queries, responses):
        # Combine query and response
        text = query + response
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            reward = reward_model(**inputs)
            rewards.append(reward.item())
    
    return torch.tensor(rewards)

def train_grpo(model, ref_model, reward_model, dataloader, config):
    """
    Main GRPO training loop
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    trainer = GRPOTrainer(model, ref_model, tokenizer, config)
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Generate responses
            queries = batch['query']
            
            # Generate multiple responses per query for group formation
            all_responses = []
            all_query_tensors = []
            all_response_tensors = []
            
            for query in queries:
                query_tensor = tokenizer.encode(query, return_tensors="pt")
                
                # Generate multiple responses
                for _ in range(config.group_size):
                    with torch.no_grad():
                        response_ids = model.generate(
                            query_tensor,
                            max_length=config.max_length,
                            do_sample=True,
                            temperature=config.temperature,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response_tensor = response_ids[0][len(query_tensor[0]):]
                    response_text = tokenizer.decode(
                        response_tensor, skip_special_tokens=True
                    )
                    
                    all_responses.append(response_text)
                    all_query_tensors.append(query_tensor[0])
                    all_response_tensors.append(response_tensor)
            
            # Compute rewards
            full_texts = [q + r for q, r in zip(queries * config.group_size, 
                                              all_responses)]
            scores = []
            for text in full_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    score = reward_model(**inputs)
                    scores.append(score.item())
            
            # Prepare batch for GRPO
            grpo_batch = {
                'query_tensors': all_query_tensors,
                'response_tensors': all_response_tensors,
                'responses': all_responses,
                'scores': scores
            }
            
            # Training step
            result = trainer.train_step(grpo_batch)
            loss = result['loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % config.log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Policy Loss: {result['policy_loss'].item():.4f}")
                print(f"KL Penalty: {result['kl_penalty'].item():.4f}")
                print(f"Mean Reward: {result['mean_reward'].item():.4f}")
        
        print(f"Epoch {epoch} completed. Average loss: {total_loss/len(dataloader):.4f}")

# Configuration
class GRPOConfig:
    lr = 1e-5
    num_epochs = 3
    group_size = 4
    beta = 0.1  # KL penalty coefficient
    max_length = 512
    temperature = 0.7
    log_interval = 10

