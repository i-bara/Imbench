### Usage

##### Build a configuration

Build a benchmark configuration in `benchmark/<config_name>.json`: 

```json
{
    "methods": ["vanilla", "ens"],
    "datasets": ["ogbn-arxiv"],
    "seeds": [100, 200]
}
```

So it will experiment in configuration of:

1. vanilla ogbn-arxiv 100

2. vanilla ogbn-arxiv 200

3. ens ogbn-arxiv 100

4. ens ogbn-arxiv 200` .

Methods are taken from `['vanilla', 'drgcn', 'smote', 'imgagn', 'ens', 'tam', 'lte4g', 'sann', 'sha', 'renode', 'pastel', 'hyperimba']`

Datasets are taken from `['Cora_100', 'Cora_20', 'CiteSeer_100', 'CiteSeer_20', 'PubMed_100', 'PubMed_20', 'chameleon_100', 'chameleon_20', 'squirrel_100', 'squirrel_20', 'Actor_100', 'Actor_20', 'Wisconsin_100', 'Wisconsin_20', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS', 'ogbn-arxiv']`

Seeds are taken from `[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`

If set to an empty array, methods / datasets / seeds will be set to all of the options.



##### Run

```python
python benchmark.py <config_name>
```

Options:

- `--gpu`: use GPU, or use CPU if not set.
- `--debug`: do not record the experiment for debug.

And the results will be recorded in `records/<config_name>(_gpu).json`.

> Warning: Do not run the same config twice at the same time!



##### Print

```python
python print.py
```

You can set what configurations to print in `print.py`.

