import matplotlib.pyplot as plt


class Draw:
    def __init__(self):
        pass

    @staticmethod
    def one_curve(X, Y, xlabel=None, ylabel=None, figsize=None, title=None, grid=False):
        assert len(X) == len(Y)
        fig = plt.figure(figsize=figsize if figsize else None)
        plt.plot(Y, X)
        if grid:
            plt.grid()

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return plt

    @staticmethod
    def two_curve(X1, Y1, X2, Y2, xlabel=None, ylabel=None, t1=None, t2=None, grid=False, figsize=None, yalign=True):
        """
        画一行两列的折线对比图
        :param X1: figure 1 标签
        :param Y1: figure 1 数据
        :param X2: figure 2 标签
        :param Y2: figure 2 数据
        :param xlabel: X 轴名，两图共享
        :param ylabel: Y 轴名，两图共享
        :param t1: figure 1 标题
        :param t2: figure 2 标题
        :param grid: 开启网格，两图一致
        :param figsize: 画布大小
        :param yalign: 是否两图 Y 轴对齐，默认 True
        :return:
        """

        assert len(X1) == len(Y1)
        assert len(X2) == len(Y2)
        fig = plt.figure(figsize=figsize if figsize else None)

        ymin = min(min(Y1), min(Y2))
        ymax = max(max(Y1), max(Y2))

        plt.subplot(1, 2, 1)
        plt.plot(X1, Y1)
        plt.title(t1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if yalign:
            plt.ylim(ymin, ymax)
        if grid:
            plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(X2, Y2)
        plt.title(t2)
        plt.xlabel(xlabel)

        if yalign:
            plt.ylim(ymin, ymax)

        if grid:
            plt.grid()

        return plt



if __name__ == "__main__":
    #fig = Draw.one_curve([1, 4, 5], ["dc", "xc", "ff"], grid=True, title="MSE Loss", ylabel="L")
    #fig.show()
    fig = Draw.two_curve(["dc", "xc", "ff"], [2,3,4], ["aa", 'v', 'df'], [2,5,7], grid=True, t1="MSE Loss", figsize=[10,5], xlabel="V", ylabel="L")
    fig.show()
    print(type(plt))