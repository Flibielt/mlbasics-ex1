from ex1.src import ex1, ex1_multi


def main():
    """
    Machine Learning Class - Exercise 1 - Linear Regression & Linear Regression with multiple
    """
    # Part 1
    # Linear Regression
    ex1()

    if input('Press ENTER to start the next part. (press [q] to exit here)\n') == 'q':
        print('Exit')
        exit(0)

    # Part 2
    # Linear Regression with multiple
    ex1_multi()


if __name__ == '__main__':
    main()
