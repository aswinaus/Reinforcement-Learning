# Reinforcement Learning

## Reinforcement Learning with Human Feedback
## Direct Policy Optimization(DPO)
## Group Relative Policy Optimization(GRPO)
Group Relative Policy Optimization (GRPO) is a specific algorithm in the family of policy optimization methods. While the exact implementation details of GRPO can vary, the core idea of policy optimization algorithms is to directly adjust the agent's policy (which is a function that maps states to actions) to maximize the expected cumulative reward.

Here's a general idea of how you would typically update the policy directly in a policy optimization algorithm like GRPO, relating it to your situation with the cosine similarity reward:

Collect Experiences: The agent interacts with the environment (in this case, this involves posing a question, the RAG system retrieving contexts and generating an answer, and calculating the cosine similarity reward) and collects a set of experiences, which usually include the state (the question), the action (the retrieval and generation process), and the reward (the cosine similarity score).

**Estimate Advantage or Gradient:** Policy optimization methods often use the concept of advantage or calculate the gradient of the expected reward with respect to the policy parameters.

Advantage: This measures how much better an action is compared to the average action in a given state. In this case, a higher cosine similarity reward for a particular retrieval/generation process would indicate a higher advantage for that "action."

**Gradient:** This indicates the direction in which to change the policy parameters to increase the expected reward.

**Update Policy Parameters:** The policy is typically parameterized by a set of weights or parameters (e.g., in a neural network). The algorithm uses the estimated advantage or gradient information to update these parameters. The goal is to shift the policy in a direction that makes actions with higher rewards (like those leading to higher cosine similarity scores) more likely to be chosen in the future. The update rule will involve the learning rate and the calculated gradient or advantage.

Repeat: Steps 1-3 are repeated over many iterations or "epochs" to iteratively improve the policy and maximize the expected cosine similarity reward.
How the Cosine Similarity Reward Fits in:

The cosine similarity reward directly serves as the signal that guides the policy update. A higher cosine similarity score for a given question and the resulting retrieved context and answer tells the GRPO algorithm that the "action" (the retrieval and generation process for that question) was good. The algorithm will then adjust its internal parameters to favor similar actions in the future when faced with similar questions.

In the context of this notebook:

Since we have calculated the cosine_similarity_reward for each question in the sample dataset, we have the necessary reward signal. To integrate this into a GRPO algorithm, we would need to:

**1.Define the policy network** (e.g., a neural network that takes the question as input and outputs the parameters for the RAG system's retrieval and generation).

**2.Defining a policy network for this example** involves creating a model that can learn to influence the RAG system's actions based on the input question to maximize the cosine similarity reward. This is a more advanced step. Here's a plan to approach this:

**3.Understand the role of the policy network:** Clarify what aspects of the RAG system the policy network will control (e.g., retrieval parameters, generation parameters, or both).

**4.Choose a suitable neural network architecture:** Select a type of neural network (e.g., a feedforward network, recurrent network, or transformer-based model) that can process the input question and output the control signals for the RAG system.

**5.Define the input and output layers:** Determine the format of the input (the question) and the output (the parameters or actions that influence the RAG system).

**6.Implement the policy network:** Write the code to define the chosen neural network architecture using a deep learning framework (like TensorFlow or PyTorch).

**7.Integrate the policy network with the RAG system:** Connect the policy network's output to the relevant parts of your RAG system so that the network's decisions influence the retrieval and generation processes.

**8.Define the training process:** Outline how the policy network will be trained using the cosine similarity reward as the optimization signal, likely involving a policy optimization algorithm like GRPO.

**9.Finish task:** Summarize the policy network definition and its role in optimizing the RAG system's performance based on the cosine similarity reward.

**Implementation:**
From [<https://colab.research.google.com/github/aswinaus/Assignments/blob/main/Agent_RewardFunction_CosineSimilarity_GroundTruth.ipynb?pli=1#scrollTo=wtwCsHlESzC0>](https://github.com/aswinaus/Reinforcement-Learning/blob/main/RAG_RewardFunction_GRPO.ipynb)

**Implement the GRPO algorithm's update rule, using the cosine_similarity_reward as the reward signal to adjust the policy network's parameters.**

Set up a training loop where repeatedly sample questions, get responses from tge RAG system, calculate the reward, and update the policy.

From <https://colab.research.google.com/github/aswinaus/Assignments/blob/main/Agent_RewardFunction_CosineSimilarity_GroundTruth.ipynb?pli=1#scrollTo=wtwCsHlESzC0>

**Weights & Biases dashboard**

**Report:**
https://api.wandb.ai/links/aswinaus-hexaware-technologies/icdaxfw5


**Source:**
https://github.com/aswinaus/Reinforcement-Learning/blob/main/RAG_RewardFunction_GRPO_W%26B.ipynb

**Highlights:**
1.Visualized the logged metrics in the Weights & Biases dashboard to analyze trends, identify correlations between metrics (e.g., reward and predicted top_k), and diagnose potential training issues such as instability or convergence problems.
2. Implement the actual RAG system reward calculation to replace the dummy reward function in the previons version of code, allowing the policy to learn based on real retrieval performance.

Yes the Avg Reward is somewhere around 0.3 which needs improvisation.

<img width="1768" height="790" alt="image" src="https://github.com/user-attachments/assets/07fb0973-717f-496e-b01a-ae9eed504dff" />

 <img width="1770" height="786" alt="image" src="https://github.com/user-attachments/assets/a611f9c5-57f6-4fb3-a028-c9931340bd38" />


The "**batch policy loss**" graph in Weights & Biases visualizes how the policy network's loss changes over batches during training. In the context of policy gradient methods, which the provided code was attempting to implement, this loss function is typically designed to encourage actions that led to higher rewards and discourage those that led to lower rewards.

Here's what the graph implies:

Decreasing Loss: A general trend of decreasing batch policy loss indicates that the policy network is learning to adjust its action probabilities (in this case, the predicted similarity_top_k) in a way that is aligned with the observed rewards. The network is becoming better at selecting actions that maximize the expected future reward.

Fluctuations: It's common to see fluctuations in the batch policy loss, especially with smaller batch sizes. This is because each batch provides a noisy estimate of the true gradient of the policy's expected reward. The policy is updated based on this noisy estimate, which can lead to variations in the loss from batch to batch.

Magnitude of Loss: The absolute value of the loss isn't as important as its trend. A large negative loss (or a small positive loss, depending on the loss function definition) generally indicates that the current policy update is leading to a significant improvement in expected reward for that batch.
Policy Updates: Each point on the graph represents the policy loss calculated for a specific batch, immediately before the policy network's weights are updated based on that loss.

In summary, the batch policy loss graph helps you monitor whether your reinforcement learning policy is learning effectively. A downward trend suggests successful learning, while a stagnant or increasing trend might indicate issues with hyperparameters, the reward function, or the network architecture.


The "**batch average reward**" graph in Weights & Biases shows the average reward obtained within each training batch. In the context of the code we were working with, the reward was calculated using cosine similarity between the generated answer from the RAG system (using the predicted similarity_top_k) and the ground truth answer for each question in the batch.

Here's what the graph implies:

Increasing Reward: A general trend of increasing batch average reward indicates that the policy network is learning to select similarity_top_k values that lead to RAG generated answers that are more similar (higher cosine similarity) to the ground truth answers. This is a positive sign, suggesting the policy is improving the RAG system's performance on your training data.

Fluctuations: Similar to the policy loss, you might see fluctuations in the batch average reward. This is natural due to the variability in the questions within each batch and the inherent randomness in the sampling of similarity_top_k from the policy network's distribution.
Magnitude of Reward: The actual value of the average reward is directly interpretable as the average cosine similarity score for the batch. A score closer to 1 indicates higher similarity and better performance for that batch.

Correlation with Loss: Ideally, as the batch average reward increases, you should see a corresponding decrease in the batch policy loss (or a trend towards a more favorable loss value, depending on the exact loss function). This is because the policy is being updated to favor actions that result in higher rewards.

**In essence, the batch average reward graph is a direct measure of your RAG system's performance on the training data under the control of the learned policy. It's a key metric to track to understand if your reinforcement learning approach is effectively improving the RAG's ability to generate relevant answers by adjusting the number of retrieved documents.
**


The "**batch average predicted top k**" graph in Weights & Biases tracks the average value of the similarity_top_k parameter that the policy network predicted for the questions within each batch during training.

Here's what this graph implies:

Policy's Action: This graph directly visualizes the action that your reinforcement learning policy is taking. The policy network outputs a mean and log-variance for a distribution (specifically, a Normal distribution in the code), and the similarity_top_k value is sampled from this distribution (and then processed to be a positive integer). The graph shows the average of these sampled and processed similarity_top_k values across a batch.

Learning Trend: As the training progresses, the trend in the "batch average predicted top k" can tell you how the policy is learning to adjust the retrieval size based on the feedback (reward) it receives.
If the average predicted top_k is increasing, it might suggest the policy is finding that retrieving more documents generally leads to better rewards for the types of questions in the training data.
If it's decreasing, the policy might be learning that retrieving fewer documents is more beneficial, perhaps because it reduces noise or improves the language model's ability to synthesize the answer.
If it fluctuates around a certain value, the policy might have converged on a preferred range for top_k.

Variability (Related to Log Variance): While this graph shows the average predicted top_k, the variability in the policy's predictions within and across batches is also important. The log-variance output of the policy network influences this variability. A decreasing trend in the log-variance (which wasn't directly logged but is an internal state of the policy) would correspond to the policy becoming more confident in its top_k predictions, potentially leading to less fluctuation in this "batch average predicted top k" graph over time.

Relationship with Reward: You should analyze this graph in conjunction with the "batch average reward" graph. The policy's goal is to adjust the predicted top_k (shown in this graph) to maximize the average reward (shown in the other graph). Observing how the trends in these two graphs correlate is crucial for understanding the learning process.

In essence, this graph provides insight into the policy network's behavior and how it's learning to control the retrieval step of your RAG system based on the training signal.


The "batch average log variance" graph in Weights & Biases shows the average value of the logarithm of the variance predicted by the policy network for the similarity_top_k action, averaged across the questions in each batch.

Here's a breakdown:

Policy Output: Our policy network outputs two values for each question: a mean and a log-variance. These define a Normal distribution from which the continuous similarity_top_k is sampled.

Variance and Log Variance: The variance (or standard deviation) represents the spread or uncertainty of this distribution. Log variance is simply the logarithm of the variance. Using log variance is common in neural networks because it ensures the predicted variance is always positive.

What the Graph Shows: The "batch average log variance" graph plots the average of these log-variance values for all the questions within a single training batch.
What the graph implies:

Policy Confidence/Exploration: The log variance is a measure of the policy's uncertainty or how much it's "**exploring**" different similarity_top_k values.

Decreasing Log Variance: A downward trend in this graph typically means the policy is becoming more confident in its predictions for similarity_top_k. It's narrowing the distribution from which it samples actions, suggesting it has found a more optimal range of top_k values for the training questions. This is generally a sign of convergence.

High or Fluctuating Log Variance: A high or fluctuating log variance might indicate that the policy is still exploring the action space (trying out a wider range of top_k values) or that it's struggling to find a consistent optimal top_k value.

Trade-off between Exploration and Exploitation: In reinforcement learning, there's a balance between **exploration** (trying new things) and **exploitation** (using what you've learned is best). **The log variance relates to exploration.** A higher log variance means more exploration (wider range of sampled top_k), while a lower log variance means more exploitation (sampling top_k values closer to the predicted mean).

In the context of our RAG training, monitoring the "batch average log variance" helps you see if your policy is settling on a preferred similarity_top_k strategy or if it's still uncertain. Ideally, you'd expect it to decrease over time as the policy learns which top_k values yield better rewards.

The "batch average advantage" graph in Weights & Biases is a crucial metric in policy gradient reinforcement learning. It shows the average "advantage" of the actions taken by the policy within each training batch.

Here's what that means:

Reward vs. Baseline: In policy gradient methods, we don't just look at the raw reward. We compare the reward obtained for a specific action to a baseline. The baseline is an estimate of the expected reward for the current state (or in our case, for the given question). It helps reduce the variance of the gradient estimates, making training more stable.

Advantage: The advantage is calculated as the difference between the actual reward received for an action and the baseline: 
                                               **Advantage = Actual Reward - Baseline.**
A positive advantage means the action taken resulted in a better-than-expected reward.
A negative advantage means the action taken resulted in a worse-than-expected reward.

Batch Average Advantage: The graph plots the average of these advantage values for all the (question, sampled top_k, reward) triplets within a single training batch.

What the graph implies:

Signal for Policy Update: The advantage is the signal used to update the policy network. The policy is trained to increase the probability of actions that had a positive advantage and decrease the probability of actions that had a negative advantage.

**Trend Towards Zero (Often):** As the policy learns and the baseline becomes a better predictor of the expected reward, the average advantage for a batch will often tend towards zero. This means the policy is consistently choosing actions that yield rewards close to what is expected, and the baseline is accurately reflecting that expectation.

Correlation with Policy Loss: The policy loss is directly calculated using the advantage. A large positive or negative average advantage for a batch will typically correspond to a larger magnitude of policy loss for that batch, as the network gets a strong signal to adjust its probabilities.

Training Stability: Monitoring the stability of the average advantage can provide insights into the training process. Extreme fluctuations might suggest an unstable baseline or issues with the reward signal.

In the context of our RAG training, the "batch average advantage" tells us how much the retrieved context (based on the policy's chosen top_k) led to a generated answer that was better or worse than the average expected answer similarity for that batch. The policy uses this signal to learn which top_k values are more likely to result in high-similarity answers.
