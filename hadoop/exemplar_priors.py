import hadoopy
from pair_hik import _find_exemplar_fn


def main():
    path = 'exemplarbank/output/1341790878.92/val_pred_pos_kern2'
    exemplars = pickle.load(open('exemplars.pkl'))

    for (kernel, row_num), columns in hadoopy.readtb(path):
        _find_exemplar_fn(exemplars[row_num][])
        


if __name__ == '__main__':
    main()
