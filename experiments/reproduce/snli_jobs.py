jobs = [
    # SGD-CE
    "python train_nli.py --opt sgd --eta 1 --loss ce --no-tqdm",

    # SGD-SVM
    "python train_nli.py --opt sgd --eta 0.1 --loss svm --no-tqdm",

    # ADAM-SVM
    "python train_nli.py --opt adam --eta 1e-4 --loss svm --no-tqdm",

    # ADAM-CE
    "python train_nli.py --opt adam --eta 1e-4 --loss ce --no-tqdm",

    # # DFW-SVM
    "python train_nli.py --opt dfw --eta 1 --loss svm --no-tqdm",
]
