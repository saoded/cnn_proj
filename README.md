# CNN Final Project
Genre Classification of Songs using CNN (2014 Academic Research Project)

## Neural Network Solution
A network of perceptrons connected together in multiple layers (hence MLP - multilayered perceptron)
Advantages:
- Modular structure
- High parallelism
- Automatic feature extraction 
- Non-Linear programming
Demands:
- Big Data & Large disk (We used 40GB)
- Large RAM (We used 24GB RAM computer)
- Strong GPU (We had NVidia GTX 660 with 960 CUDA cores)

### MLP techniques
Layers:
- Convolution layer (CNN) - proved itself with images, we wanted to test it on sound
- Max pooling - subsampling to reduce and simplify the data
- Dropout - remove connections randomly to train only parts of the network at different time, for regularization
- Batch normalization - for convergence speedup. Proved very useful
- Non-linear element: ReLU - for efficient gradient propagation
- Log-Softmax - final layer, calculate matching rate between a song and the genre/artist, take best-matching genre
Optimization:
- Log likelihood
- Backpropagation & SGD
- Momentum - prevents from converging in local minima
- LR decay - faster partial convergence fitting for the epoch model (each repetition starts from a better point then it predecessor)
- Weigth decay - used for regularization. Penalize “complicated” solutions

## The Data
A DB of over 10 thousand songs. divided to 90% training set and 10% testing set. test set is expanded with songs by artists that aren't in the training set
- Number of genres, M=5
- Number of Artists, N=15 (3 per genre to avoid bias)
-- It was difficult to find a DB with genre tags per song, we setteled for genre tag per artist (influences the netwiork)
- Songs per artist, K (avoid bias toward artist)

### The Input
This projects contains a comparison between using MFCC as input, a standard audio feature set, and using raw wav data as input.
since MFCC are 16 coefficients assinged to each audio interval, a fair comparison would mean adding to the raw-input network a first layer with 16 units

### The Output
- The classified genre
- Artist classification, used for improving the genre classification (20% weight)
- Feature vector for each interval

### Environment
- [Torch7](http://torch.ch/), a luajit framework for neural networks and N-dimensional arrays (tensors), efficiently utilizing the GPU if available. Runs natively on Linux systems.
- Input batching for performance gain (parallelism utilization)

### Results
- 96.67% correctness on the test set genre
- An equivalent CNN using MFCC as preprocessing gave way worse results, 70.7% correctness
- Surprisingly, increasing the number of genres improved correctness rate, as the machine got better understanding of the problem in general (by learning more nuanced features)
