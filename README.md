# VAE-GANMDA：A Microbe-Drug Association Prediction Model Integrating Variational Autoencoders and Generative Adversarial Networks

# VAE-GANMDA Overall Architecture Diagram
![image](https://github.com/user-attachments/assets/c83d7910-d835-42f6-8620-968aab72ee59)

VAE-GANMDA overall workflow: 
Step 1: Integrate multi-source information regarding microbes and drugs, incorporating their similarities to construct an associated feature network. 
Step 2: Train and fine tune the VAE and GAN fusion model using relevant feature networks, and use CBAM to enhance the extraction of non-linear manifold features. 
Step 3: Extract linear features from the data using the SVD decomposition technique. 
Step 4: Apply k-means++ for selecting high-quality negative samples. 
Step 5: Train the MLP, ultimately predicting unknown microbe-drug associations.

# Experimental Environment
python = 3.9
torch = 2.4.0+cu121
matplotlib = 3.9
numpy = 1.26
scikit-learn = 1.5

# Experimental equipment
processing unit： Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz 2.50 GHz

# Experimental Result
![image](https://github.com/user-attachments/assets/9956875d-c794-4337-8c2d-f9b46a62fa10)
