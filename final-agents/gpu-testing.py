import torch as T

def gpu_test():
    if T.cuda.is_available():
        print("GPU!")
    else:
        print("No GPU!")

if __name__ == "__main__":
    gpu_test()