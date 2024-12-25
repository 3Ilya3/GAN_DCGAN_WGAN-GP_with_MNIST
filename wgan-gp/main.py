import wgan_gp as g
from data import mnist_dataloader

def main():
    lr = 0.0002
    gan = g.WGAN_GP(lr)
    gan.train(mnist_dataloader, num_epochs=1000)
    
if __name__ == "__main__":
    main()
