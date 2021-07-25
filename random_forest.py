'''
                            _ooOoo_
                           o8888888o
                           88" . "88
                           (| -_- |)
                           O\  =  /O
                        ____/`---'\____
                      .'  \\|     |//  `.
                     /  \\|||  :  |||//  \
                    /  _||||| -:- |||||-  \
                    |   | \\\  -  /// |   |
                    | \_|  ''\---/''  |   |
                    \  .-\__  `-`  ___/-. /
                  ___`. .'  /--.--\  `. . __
               ."" '<  `.___\_<|>_/___.'  >'"".
              | | :  `- \`.;`\ _ /`;.`/ - ` : | |
              \  \ `-.   \_ __\ /__ _/   .-` /  /
         ======`-.____`-.___\_____/___.-`____.-'======
                            `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      Buddha Bless, No Bug !
'''

from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np



def fenlei():
    iris = load_iris()
    train_feature,test_feature,train_label,test_label = train_test_split(iris.data,
                                                                         iris.target,
                                                                         test_size=0.2,
                                                                         random_state=100)

    tree = DecisionTreeClassifier(random_state=0)
    trees = RandomForestClassifier(n_estimators=100,
                                   criterion='gini',
                                   max_features='auto',
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0,
                                   bootstrap=True,
                                   random_state=0
                                   )
    '''
    n_estimators:森林里的决策树数目
    criterion:分裂的依据（函数），基尼指数或信息熵增益
    max_features:寻找最佳分割时需要考虑的特征数目
    max_depth:树的最大深度
    min_samples_split:分割内部节点所需要的最小样本数量
    min_samples_leaf:叶子结点的最小样本数量
    min_weight_fraction_leaf:没看懂
    max_leaf_nodes:最大的叶子结点数量
    min_impurity_decrease:没看懂
    bootstrap:建立决策树时，是否使用有放回抽样
    '''
    tree.fit(train_feature, train_label)
    trees.fit(train_feature,train_label)
    trees_score = trees.score(test_feature,test_label)
    tree_score = tree.score(test_feature,test_label)
    print("决策树分类准确率为：{}".format(tree_score))
    print("随机森林分类准确率为：{}".format(trees_score))

def huigui():
    boston = load_boston()
    train_feature,test_feature,train_label,test_label = train_test_split(boston.data,
                                                                         boston.target,
                                                                         test_size=0.2,
                                                                         random_state=1)
    tree = DecisionTreeRegressor(random_state=0)
    trees = RandomForestRegressor(n_estimators=100,
                                   criterion='mse',
                                   max_features='auto',
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0,
                                   bootstrap=True
                                   )
    '''
    回归于分类器区别只在于节点分裂的函数依据。
    '''
    tree.fit(train_feature, train_label)
    trees.fit(train_feature,train_label)
    trees_score = mean_squared_error(trees.predict(test_feature),test_label)
    tree_score = mean_squared_error(tree.predict(test_feature),test_label)
    print("决策树回归均方误差为：{}".format(tree_score))
    print("随机森林回归均方误差：{}".format(trees_score))
def main():
    print("--分类--")
    fenlei()
    print("--回归--")
    huigui()

if __name__ == '__main__':
    print("--入口--\n")
    main()
