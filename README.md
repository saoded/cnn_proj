# cnn_proj
Genre Classification of Songs using CNN (2014 Academic Research Project)

Neural Network Solution
A network of perceptrons connected together in several layers
Advantages:
- Modular structure
- High parallelism
- Automatic feature extraction 
- Non-Linear programming
Demands:
- Big Data & Large disk (We used 40GB)
- Large RAM (We used 24GB RAM computer)
- Strong GPU (We had 960 CUDA cores)

The Data
A DB of over 10 thousand songs.
- Number of genres, M=5
- Artists per genre, N/M (avoid bias toward genre)
- Songs per artist, K (avoid bias toward artist)
- 10sec intervals per song, L
