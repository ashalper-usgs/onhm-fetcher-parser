from multiprocessing import Process

# See
# https://docs.python.org/2/library/multiprocessing.html#the-process-class

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
