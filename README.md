# DeepB<sup>3</sup>P
DeepB<sup>3</sup>P: A Transformer-Based Model for Identifying and Generating Blood-Brain Barrier Penetrating Peptides by Using Feedback GAN<br/>
The feedback generative adversarial network (FBGAN) model was employed to effectively generate analogous BBBPs, addressing data imbalance.
<div align="center"> <img src="https://github.com/GreatChenLab/deepB3P/assets/90399926/bc2d32ee-d456-4827-9fff-c5dc892a28dd" width="55%"></div>

# Environment
1. The model training and testing process was performed on a CentOS Linux system. The CPU was an Intel(R) Xeon(R) Gold 62336Y CPU @ 2.40 GHz, which had 48 logical CPUs. The GPU was the NVIDIA A100, which had a memory capacity of 80G.
2. The packages used for deepB<sup>3</sup>P are available in environment.yml and requirements.txt, and can be installed using conda install and pip install respectively.
# Test deepB<sup>3</sup>P on new data
`python predict_user.py fasta_file` <br/>
***The input file must be in fasta format***<br/>
The output file is prob.csv in the current directory, containing the input sequences and predicted probabilities
# Test FBGAN-based model
`python  fbgan_model.py`<br/>
1.  The number of sequences output can be made active by setting the n_sequences parameter in the config_fbgan.py file
2.  if n_epochs != 1,000, then train a new FBGAN model, and user can modify the parameter for training a new model in config_fbgan.py
3.  the output file is in the fbgan/out

# Explanation of the main parameters

 | parameter        | Explanation   | 
| --------   | -----:  |
|seq_len |the max legth of peptide, default 50 |
|vocab_size |for train fbgan, 20 aa add 1 'X', default 21|
|drop |dropout for all models, default 0.3|
|bs |batch size, default 16|
|n_epochs |epochs for all models, default 200|
|lr |learning rate, default 0.0001|
|kFold |k-fold cross validation, default 5|
|earlyStop |Early stops the training if validation acc doesn't improve after a given patience, default 5|
|reload |Whether to start training based on an existing model，default True|
|is_blast |Whether to use the feedback function (BLAST)，default True|
|d_model |embedding size for deepB<sup>3</sup>P，default 64|
|d_ff |feedforeard dimension size for deepB<sup>3</sup>P，default 16|
|d_k |dimension of K(=Q), V  for deepB<sup>3</sup>P，default 32|
|n_layers |number of encoder of decoder layer  for deepB<sup>3</sup>P，default 1|
|n_heads |number of head  for deepB<sup>3</sup>P，default 2|

