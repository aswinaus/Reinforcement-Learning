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

**Source:**
https://github.com/aswinaus/Reinforcement-Learning/blob/main/RAG_RewardFunction_GRPO_W%26B.ipynb

**Highlights:**
1.Visualized the logged metrics in the Weights & Biases dashboard to analyze trends, identify correlations between metrics (e.g., reward and predicted top_k), and diagnose potential training issues such as instability or convergence problems.
2. Implement the actual RAG system reward calculation to replace the dummy reward function in the previons version of code, allowing the policy to learn based on real retrieval performance.

Yes the Avg Reward is somewhere around 0.3 which needs improvisation.

<img width="1768" height="790" alt="image" src="https://github.com/user-attachments/assets/07fb0973-717f-496e-b01a-ae9eed504dff" />

 <img width="1770" height="786" alt="image" src="https://github.com/user-attachments/assets/a611f9c5-57f6-4fb3-a028-c9931340bd38" />


