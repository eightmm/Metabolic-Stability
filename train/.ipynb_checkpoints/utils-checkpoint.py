import dgl, random, numpy, torch

from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

def format_pytorch_version(version): return version.split('+')[0]

def format_cuda_version(version): return 'cu' + version.replace('.', '')

def format_pyg_version(version): return version

def format_dgl_version(version): return version

def set_random_seed(seed=42):
    dgl.seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[INFO] RANDOM, DGL, NUMPY and TORCH random seed is set {seed}.')

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params:,}")

def analysis(pred, true):

    try:
        roc_auc    = roc_auc_score(true, pred)
    except:
        roc_auc    = 0

    true = [ 1 if i > 0.5 else 0 for i in true ]
    pred = [ 1 if i > 0.5 else 0 for i in pred ]

    binary_acc     = accuracy_score(true, pred)
    precision      = precision_score(true, pred)
    recall         = recall_score(true, pred)
    f1             = f1_score(true, pred)

    mcc            = matthews_corrcoef(true, pred)
    TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
    sensitivity    = 1.0 * TP / (TP + FN)
    specificity    = 1.0 * TN / (FP + TN)
    NPV            = 1.0 * TN / (TN + FN)

    result = {
        'ACC': binary_acc,
        'MCC': mcc,
        'Sensitivity Recall': sensitivity,
        'Specificity': specificity,
        'Precision PPV': precision,
        'NPV': NPV,
        'F1': f1,
        'ROC_AUC': roc_auc,
    }
    return result


def analysis_table(dct, type='float'):
    table = PrettyTable()
    for c in dct.keys():
        table.add_column(c, [])
    if type == 'int':
        table.add_row([ f'{int(c)}' for c in dct.values()])
    else:
        table.add_row([ f'{c:.3f}' for c in dct.values()])
    print(table)
