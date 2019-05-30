# Models Directory
Currently contains best-performing transfer-learned model (best perplexity: ~57).
Architecture consists of:

* 500-dimensional embeddings learned from s2s-training on a different, larger corpus
* 2 unidirectional encoder LSTM's
* 1 unidirectional decoder LSTM
* Simple dot-product based attention
