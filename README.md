# Gremlin Algorithm written in JAX



### Install
```
pip install -r requirements.txt

```


## Available flags:
```
 -msa ####.fa              | multi-line fasta format multiple sequence alignment
 -gaps [True]/False        | [retain]/eliminate gaps from mrf calculation
 -filter_msa True/[False]  | remove sequences that are more than 25% gap and columns that are more than 75% gaps
 -filter_msa_row [0.25]    | remove sequences that are more than [x]% gap
 -filter_msa_col [0.75]    | remove  columns  that are more than [x]% gaps

```

Produce predictions of which residues are in contact and view the probability of each amino acid at each position based on the Markov Random Field.

## Example run:
`python GREMLIN_JAX.py -msa 2lx7.sto -gaps -filter_msa`

| Predicted Contact Map | Amion Acid Frequency|
| ---------------------- | -------------------------- |
|![](/img/2oug_cmap.png) | ![](/img/2oug_aa_freq.png) |
|![](/img/2lx7_cmap.png) | ![](/img/2lx7_aa_freq.png) |


## Other approaches to Coevolution
- [GremlinCPP](https://github.com/sokrypton/GREMLIN_CPP) Gremlin Algorithm written in C++
- [Direct Coupling Analysis](https://github.com/KIT-MBS/pydca) A mean-field approach to computing coevolutionary contacts
- [MSATransformer](https://github.com/rmrao/msa-transformer) Unsupervised protein language model that can preict contacts


