# HeartNet

HeartNet is a modular, domain-agnostic computational pipeline designed for the clustering of multilayer networks based on structural similarities. HeartNet is introduced in the Bachelor Thesis included in this repository.

This repository contains the implementations of the individual (modular and domain-agnostic) stages of HeartNet and a domain-specific script `heartnet.py` which comprises the individual stages of HeartNet for the clustering of heart tissue samples into zones of ischemic heart disease. This script may be used as a template for the generalization of HeartNet to other applications.

HeartNet is highly modular and is configured via a single TOML configuration file.


## Installation
To install HeartNet, clone this repository by running
```
$ git clone git@github.com:RobertGiesler/HeartNet.git
```

From the root of the repository, then run
```
$ pip install .
```

## Usage
To run HeartNet with the configuration file `config.toml`, run the following command from the root of the repository:
```
$ python3 -m src.heartnet.heartnet --config <config_path>
```
where `<config_path>` is the path of `config.toml`.

### Excluding Spatial Information
To run HeartNet only on the gene expression layers of the samples with the configuration file `config.toml`, run the following command from the root of the repository:
```
$ python3 -m src.heartnet.heartnet_celltype_only --config <config_path>
```

### Visualization of Results
The Jupyter Notebook `visualizations.ipynb` provides cells for the visualization of the resulting Gromov-Wasserstein distance matrices, clustermaps, and dendrograms.

## Configuration
HeartNet is configured via a single TOML configuration file. A template configuration file `sample_config.toml` is included in this repository. The configuration parameters of HeartNet are summarized in the following.

### Configuration Parameters
`processes` (`int`): Number of CPUs available for multiprocessing.

#### `data`
`raw_data` (`str`): Path of the directory containing the data of the multilayer networks to be clustered. Each multilayer network is contained within a subdirectory and is stored in the form of adjacency matrices as CSV files. The subdirectory of each multilayer network must contain an adjacency matrix for each layer and an inter-layer connection matrix for each pair of layers which are connected by inter-layer edges. If the network is directed, two inter-layer connection matrices are required for each layer pair - one for each direction. Layer adjacency matrices must be named according to the following pattern: `*_layer_X.csv` where `X` is the name of the layer. Inter-layer connection matrices must be named according to the following pattern: `*_interlayer_X_Y.csv` where `X` is the name of the layer which the nodes on the rows of the CSV file belong to and `Y` is the name of the layer which the nodes on the columns of the CSV file belong to. Avoid `_` in the layer names.

`preprocessed_data` (`str`): Path of the parent directory where the multilayer network adjacency matrices are stored after sparsification and normalization. If the configuration defines a combination of sparsification and normalization settings that has not previously been used, a new subdirectory for this combination is created. If this combination has previously been used, the subdirectory created for the previous run is reused.

`edgelists` (`str`): Path of the parent directory where the multilayer edge lists are stored after sparsification and normalization. If this combination of sparsification and normalization settings has been previously used, the edge list subdirectory created for that run is reused.

`embeddings` (`str`): Path of the parent directory where the node embeddings of are stored in CSV format. For each HeartNet run, a new subdirectory with a name of the form `<config_name>_YYYY-MM-DDTHHMMSS` is created.

`gw_dir` (`str`): Path of the parent directory where the Gromov-Wasserstein distanced between samples are stored in PKL format. For each HeartNet run, a new subdirectory with a name of the form `<config_name>_YYYY-MM-DDTHHMMSS` is created.

`clusterings` (`str`): Path of the parent directory where the clustering results are stored in CSV format. For each HeartNet run, a new subdirectory with a name of the form `<config_name>_YYYY-MM-DDTHHMMSS` is created.

#### `preprocessing`
`weights` (`'distance'` or `'similarity'`): Whether the edge weights of the network represent distance or similarity between nodes.

`sparsify_spot` (`bool`): Whether to sparsify the spatial layer.

`spot_k` (`int > 0`): The `k` used in KNN sparsification of the spatial layer. Only applies if `sparsify_spot == True`.

`sparsify_celltype` (`bool`): Whether to sparsify the gene expression layer.

`celltype_k` (`int > 0`): The `k` used in KNN sparsification of the gene expression layer. Only applies if `sparsify_celltype == True`.

`sparsify_interlayer` (`bool`): Whether to sparsify the inter-layer edges.

`interlayer_method` (`'knn'` or `'threshold'`): Whether to use KNN or threshold sparsification for the interlayer edges.

`interlayer_k` (`int > 0`): The `k` used in KNN sparsification of the inter-layer edges. Only applies if `sparsify_interlayer == True` and `interlayer_method == 'knn'`.

`interlayer_stds` (`int`): The number of standard deviations away from the mean edge weight where to set the threshold value. Only applies if `sparsify_interlayer == True` and `interlayer_method == 'threshold'`.

`normalization.spot` (`bool`): Whether to normalize the edges in the spatial layer.

`normalization.celltype` (`bool`): Whether to normalize the edges in the gene expression layer.

`normalization.interlayer` (`bool`): Whether to normalize the inter-layer edges.

#### `embedding`
`algorithm` (`str`): The node embedding algorithm used. Currently, the only option is `'HeNHoE-2vec'`.

##### `parameters`
`p` (`float > 0`): Return parameter of the HeNHoE-2vec algorithm.

`q` (`float > 0`): In-out parameter of the HeNHoE-2vec algorithm.

`s` (`float > 0` or list of dicts): The switching parameter of the HeNHoE-2vec algorithm. If `s` is a float, the same switching parameter is used for all pairs of layers. Otherwise, different switching parameters can be defined for different pairs of layers, as exemplified in the following:

```python
s = [
    {source = "celltype", target = "spot", s = 3},
    {source = "spot", target = "celltype", s = 1}
]
```

`dims` (`int > 0`): The dimensionality of the node embeddings.

`num_walks` (`int > 0`): The number of random walks sampled per node.

`walk_length` (`int > 0`): The length of each random walk.

`window_size` (`0 < int <= walk_length`): The window size used by the Skip-gram model to learn the node embeddings.

`epochs` (`int > 0`): The number of epochs for which to train the Skip-gram model.

#### `gromov_wasserstein`
`calculcate_gw` (`bool`): Whether to calculate the *unregularized* Gromov-Wasserstein distance between node embeddings.

`calculate_entr_gw` (`bool`): Whether to calculate the entropically *regularized* Gromov-Wasserstein distance between node embeddings.

`epsilon` (`float > 0`): The regularization weight of the entropically regularized Gromov-Wasserstein optimal transport problem. Only applies if `calculate_entr_gw == True`.

`distribution` (`'uniform'` or `'balanced'`): Whether the probaility distributions of the Gromov-Wasserstein optimal transport problem shall be uniform of balanced between layers (see Thesis for more details).

`alpha` (`0 <= float <= 1`): Balancing parameter. Only applies if `distribution == 'balanced'`.

#### `clustering`
`linkage_method` (`'single'`, `'complete'`, `'average'`, or `'ward'`): Linkage method used in agglomerative hierarchical clustering.

`n_clusters` (`int > 0`): Number of clusters to be yielded by the clustering.
