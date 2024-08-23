<!DOCTYPE html>
<html>
<head>

</head>
<body>
<h2>Intent Aware Recommender Systems</h2>
<p align="center">
  <img src="intentAware.webp" width="300", title="Intent Aware Recommender Systems">
  
</p>


<h3>Introduction</h3>
<p align="justify">This reproducibility package was prepared for the paper titled "Performance Comparison of Intent Aware and Non-Intent Aware Recommender Systems" and submitted to the ABC.  The results reported in this paper were achieved with the help of the codes, which were shared by the original authors of the selected articles. For the implementation of baseline models, we utilized the session-rec and RecSys2019_DeepLearning_Evaluation  frameworks. These frameworks include the state-of-the-art baseline models for session based and top-n recommender systems. More information about the session-rec and RecSys2019_DeepLearning_Evaluation frameworks can be found by following the given links. </p>
<ul>
  <li><a href="https://rn5l.github.io/session-rec/index.html" target="_blank">Session rec framework</a></li>
  <li><a href="https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation.git" target="_blank"> RecSys2019_DeepLearning_Evaluation  framework </a></li>
</ul>
<h5>Selected articles</h5>

<ul>
  <li>STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation (KDD'2018)</li>
  <li>Neural Attentive Session-based Recommendation (SIGIR'2018)</li>
  <li>TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation (SIGIR'2020)</li>
  <li>GCE-GNN: Global Context Enhanced Graph Neural Networks for Session-based Recommendation (SIGIR'20)</li>
  <li>Enhancing Hypergraph Neural Networks with Intent Disentanglement for Session-based Recommendation (SIGIR'2022)</li>
  <li>Dynamic Intent Aware Iterative Denoising Network for Session-based Recommendation (Journal: Information Processing & Management'2022 - IF: 7.4)</li>  
  <li>Disentangled Graph Collaborative Filtering (SIGIR'2020)</li>
  <li>Learning Intents behind Interactions with Knowledge Graph for Recommendation (WWW'2021) </li>
  <li>Intent Disentanglement and Feature Self-Supervision for Novel Recommendation (Journal: IEEE Transactions on Knowledge and Data Engineering'2022 - IF: 8.9) </li>

</ul>
<h5>Required libraries to run the framework</h5>
<ul>
  <li>Anaconda 4.X (Python 3.5 or higher)</li>
  <li>numpy=1.23.5</li>
  <li>pandas=1.5.3</li>
  <li>torch=1.13.1</li>
  <li>scipy=1.10.1</li>
  <li>python-dateutil=2.8.1</li>
  <li>pytz=2021.1</li>
  <li>certifi=2020.12.5</li>
  <li>pyyaml=5.4.1</li>
  <li>networkx=2.5.1</li>
  <li>scikit-learn=0.24.2</li>
  <li>keras=2.11.0</li>
  <li>six=1.15.0</li>
  <li>theano=1.0.3</li>
  <li>psutil=5.8.0</li>
  <li>pympler=0.9</li>
  <li>Scikit-optimize</li>
  <li>tensorflow=2.11.0</li>
  <li>tables=3.8.0</li>
  <li>scikit-optimize=0.8.1</li>
  <li>python-telegram-bot=13.5</li>
  <li>tqdm=4.64.1</li>
  <li>dill=0.3.6</li>
  <li>numba</li>
</ul>
<h2>Installation guide</h2>  
<p>This is how the framework can be downloaded and configured to run the experiments</p>
  
<h5>Using Docker</h5>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to "pull Docker Image" from Docker Hub: <code>docker pull shefai/intent_aware_recomm_systems</code>
  <li>Clone the GitHub repository by using the link: <code>https://github.com/Faisalse/Intent_Aware_Recomm_Systems.git</code>
  <li>Move into the <b>Intent_Aware_Recomm_Systems</b> directory</li>
  
  <li>Run the command to mount the current directory <i>Intent_Aware_Recomm_Systems</i> to the docker container named as <i>intent_aware_recomm_systems_container</i>: <code>docker run --name intent_aware_recomm_systems_container  -it -v "$(pwd):/Intent_Aware_Recomm_Systems" -it shefai/intent_aware_recomm_systems</code>. If you have the support of CUDA-capable GPUs then run the following command to attach GPUs with the container: <code>docker run --name intent_aware_recomm_systems_container  -it --gpus all -v "$(pwd):/SessionRecGraphFusion" -it shefai/intent_aware_recomm_systems</code></li> 
<li>If you are already inside the runing container then run the command to navigate to the mounted directory <i>Intent_Aware_Recomm_Systems</i>: <code>cd /Intent_Aware_Recomm_Systems</code> otherwise starts the "intent_aware_recomm_systems_container"</li>
<li>Finally, follow the given instructions to run experiments for each model </li>
</ul>  


<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/Faisalse/Intent_Aware_Recomm_Systems.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>Intent_Aware_Recomm_Systems</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name Intent_Aware_Recomm_Systems python=3.8</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate Intent_Aware_Recomm_Systems</code></li>
    <li>Run this command to install the required libraries for CPU: <code>pip install -r requirements_cpu.txt</code>. However, if you have support of CUDA-capable GPUs, 
        then run this command to install the required libraries to run the experiments on GPU: <code>pip install -r requirements_gpu.txt</code></li>
    <li>If you do not understand the instructions, then check the video to run the experiments: (https://youtu.be/uCW2omAxYP8?si=UW_YjJ_GqACuc_Gs)</li>
  </ul>
</p>

<h2>Follow these steps to reproduce experiments for Intent Aware and Non-Intent Aware Recommender Systems</h2>
<h5>TAGNN and baseline models</h5>
<ul>
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Yoochoose</a> dataset, unzip it and put the “yoochoose-clicks.dat” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the TAGNN and baseline models on the shorter version of the Yoochoose dataset: <code>python run_experiments_TAGNN_And_baseline_models.py --dataset yoochoose1_64</code></li>
  
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Diginetica</a> dataset, unzip it and put the “train-item-views.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the TAGNN and baseline models on the Diginetica dataset: <code>python run_experiments_TAGNN_And_baseline_models.py --dataset diginetica</code></li> 
</ul>

<h5>GCE_GNN and baseline models</h5>
<ul>
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Diginetica</a> dataset, unzip it and put the “train-item-views.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the GCE_GNN and baseline models on the Diginetica dataset: <code>python run_experiments_GCE_GNN_And_baseline_models.py --dataset diginetica</code></li> 

<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Nowplaying</a> dataset, unzip it and put the “nowplaying.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the GCE_GNN and baseline models on the Nowplaying dataset: <code>python run_experiments_GCE_GNN_And_baseline_models.py --dataset nowplaying</code></li> 

<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Tmall</a> dataset, unzip it and put the “dataset15.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the GCE_GNN and baseline models on the Tmall dataset: <code>python run_experiments_GCE_GNN_And_baseline_models.py --dataset tmall</code></li> 
</ul>

<h5>DIDN and baseline models</h5>
<ul>

<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Yoochoose</a> dataset, unzip it and put the “yoochoose-clicks.dat” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the TAGNN and baseline models on the shorter version of the Yoochoose dataset: <code>python run_experiments_for_DIDN_baseline_models.py --dataset yoochoose1_64</code> and run the following command to create the experiments for the larger version of the Yoochoose dataset <code>python run_experiments_for_DIDN_baseline_models.py --dataset yoochoose1_4</code>  </li>
  
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Diginetica</a> dataset, unzip it and put the “train-item-views.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the DIDN and baseline models on the Diginetica dataset: <code>python run_experiments_for_DIDN_baseline_models.py --dataset diginetica</code></li> 

</ul>

<h5>NARM and baseline models</h5>
<ul>

<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Yoochoose</a> dataset, unzip it and put the “yoochoose-clicks.dat” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the NARM and baseline models on the shorter version of the Yoochoose dataset: <code>python run_experiments_for_NARM_And_baseline_models.py --dataset yoochoose1_64</code> and run the following command to create the experiments for the larger version of the Yoochoose dataset <code>python run_experiments_for_NARM_And_baseline_models.py --dataset yoochoose1_4</code>  </li>
  
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Diginetica</a> dataset, unzip it and put the “train-item-views.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the DIDN and baseline models on the Diginetica dataset: <code>python run_experiments_for_NARM_And_baseline_models.py --dataset diginetica</code></li> 

</ul>

<h5>HIDE and baseline models</h5>
<ul>

<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Tmall</a> dataset, unzip it and put the “dataset15.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the HIDE and baseline models on the Tmall dataset: <code>python run_experiments_HIDE_And_baseline_models.py --dataset tmall</code></li> 
<li>Run this command to reproduce the experiments for the HIDE model with original train-test splits and without any modification in the code: <code>python run_experiments_for_HIDE_withoutAnyChanges.py --dataset tmall</code></li>
</ul>
<h5>KIGN and baseline models</h5>
<ul>
<li>Run this command to reproduce the experiments for the KGIN and baseline models on the lastFm dataset: <code>python run_experiments_for_KGIN_baselines_algorithms.py --dataset lastFm</code>  </li>

<li>Run this command to reproduce the experiments for the KGIN and baseline models on the alibabaFashion dataset: <code>python run_experiments_for_KGIN_baselines_algorithms.py --dataset alibabaFashion</code>  </li>
<li>Run this command to reproduce the experiments for the KGIN and baseline models on the amazonBook dataset: <code>python run_experiments_for_KGIN_baselines_algorithms.py --dataset amazonBook</code>  </li>
</ul>
<h5>STAMP and baseline models</h5>
<ul>
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Yoochoose</a> dataset, unzip it and put the “yoochoose-clicks.dat” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the STAMP and baseline models on the shorter version of the Yoochoose dataset: <code>python run_experiments_STAMP_And_baseline_models.py -m stamp_rsc -d rsc15_64 -n</code> and run the following command to create the experiments for the larger version of the Yoochoose dataset <code>python run_experiments_STAMP_And_baseline_models.py -m stamp_rsc -d rsc15_4 -n</code>  </li>
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Diginetica</a> dataset, unzip it and put the “train-item-views.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the STAMP and baseline models on the Diginetica dataset: <code>python run_experiments_STAMP_And_baseline_models.py -m stamp_cikm -d digi -n</code></li> 
</ul>


</body>
</html>  

