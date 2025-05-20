# DA6401-Assignment-3
Assignment 3 of DA6401, Introduction to Deep Learning 

Report: https://wandb.ai/alandandoor-iit-madras/DL_A3/reports/Assignment-3--VmlldzoxMjc4Nzc4Ng?accessToken=lwd1ytj30khb5sb5x6c2iixlsrih8bzvmtuksnxusguno0q1xw49aqbk72s0x1fk

Github: https://github.com/AndoorAlanD/DA6401-Assignment-3

**Codes**
-----------------
**DL_As3_Q1-4.py** :- This python file contains the code for questions 1 to 4 of assignment and can be called from command line arguments as shown below.
-python DL_As3_Q1-4.py
-python DL_As3_Q1-4.py --cell_type rnn --inp_embed_size 128 --hidden_size 256 --batch_size 64 --lr 0.0001 --dropout 0.2 --num_enc_layers 2 --num_dec_layers 2

**DL_As3_Q1-4.ipynb** :- This the collab version of 'DL_As3_Q1-4.py' and in this you can find the different sweep configurations used and its result.

**DL_As3_Q5-6.py** :- This python file contains the code for questions 5 and 6 of assignment and can be called from command line arguments as shown below.
-python DL_As3_Q5-6.py
-python DL_As3_Q5-6.py --cell_type gru --hidden_dim 128 --batch_size 64 --dropout 0.2 --lr 0.0001

**DL_As3_Q5-6.ipynb** :-  This the collab version of 'DL_As3_Q5-6.py' and in this you can find the different sweep configurations used and its result. To run this in colab/kaggle you will have to upload the 'Noto_Serif_Malayalam' font and set the path for it properly in the 1st cell itself.

**Noto_Serif_Malayalam** :- This is a font type that i used as malayalam words where not correctly getting printed in the heatmap. Be sure to upload and use it when needed.

Note: For all codes be sure to change the dataset location before running the code.
