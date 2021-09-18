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
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn.modules as nn

class Linear_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d()
        self.Linear_1 = nn.Linear(13,6)
        self.sig = nn.Sigmoid()
        self.Linear_2 = nn.Linear(6,3)
    def forward(self,input):
        x = self.Linear_1(input)
        x = self.sig(x)
        x = self.Linear_2(x)
        x = torch.softmax(x, dim=1)
        return x
def main():
    wine = datasets.load_wine()
    train_feature, test_feature, train_label, test_label = train_test_split(wine.data,wine.target,test_size=0.2,random_state=100)



if __name__ == '__main__':
    print("--入口--\n")
    main()
