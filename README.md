# kinase_deep_learning_app
a flask application to predict kinase inhibition across 342 human kinases using multitask deep learning

This application predicts the probability of inhibition for molecules specified by a smiles string using a multitask deep neural network trained using over 700,000 human kinase bioactivity annotations for over 300,000 unique small molecules from ChEMBL and the Kinase Knowledge Base across 342 kinases. Each smiles string should be separated by a new line with no delimiter.
