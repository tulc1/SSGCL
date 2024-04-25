### 1. Note directories

Before running the codes, please ensure that two directories `log/` and `saved_model/` are created under the root directory. They are used to store the training results and the saved model and optimizer states.

### 2. Running environment

We develope our codes in the following environment:

```
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

### 3. How to run the codes

* Yelp
```
python main.py --lr 5e-5 --lambda1 0.4 --temp1 0.2 --lambda2 0.5 --temp2 0.2 --reg 1e-3 --alpha 1. --hyper 32 --gnn_layer 2 --data yelp
```

* Gowalla

```
python main.py --lr 1e-5 --lambda1 0.4 --temp1 0.2 --lambda2 1.0 --temp2 1.0 --reg 1e-3 --alpha 3. --hyper 128 --gnn_layer 2 --data gowalla
```

* Tmall
* 
```
python main.py --lr 5e-5 --lambda1 0.8 --temp1 0.2 --lambda2 1.0 --temp2 2.0 --reg 1e-3 --alpha 2. --hyper 128 --gnn_layer 2 --data tmall
```


