#@author: limengqi
#@date: 2024-12-02
"""
Implementation of SaLoRA(SAFETY-ALIGNMENT PRESERVED LOW-RANK ADAPTATION), 
a initialize method for lora, based on paper: https://openreview.net/pdf?id=GOoVzE9nSj.

In SaLoRA:

Here for consistency in the repo, not using the same notations as in the paper:

W_0 = W_{res} = W - CA_0B_0^T \in \mathbb{R}^{m \times n},
where A_0, B_0 are the top r singular vectors of the output features WX_h

C = I - U_cU_c^T, where U_c is the top r_s singular vectors of the output features WX_h
on harmful prompts and the paper call r_s as the safety rank.

W = \bar{U}\bar{S}\bar{V}

A_0 = UU^T \bar{U}[:,:r] \sqrt{\bar{S}[:r,:r]}

B_0 = \sqrt{\bar{S}[:r,:r]} \bar{V}^T[:,:r]

where U \in \mathbb{R}^{m \times r} is the top r_t left singular vectors of WX_t.

X_t here denotes the input features of LLM's layers concerning input samples of certain downstream tasks.

the experiments the paper performed is based on r_s = r_t = 32

Here, for simplicity we assume X_h = X_t (Is this right?)

Moreover, the paper adopt four state-of-the-art **post-hoc** safety alignment methods
on LoRA fine-tuned models as baselines while here we dont try to replicate them
"""
