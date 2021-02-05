import torch as T

if __name__ == '__main__':
    if T.cuda.is_available():
        print('gpu')
    else:
        print('cpu')
    #T.device('cuda:0' if T.cuda.is_available() else 'cpu')
