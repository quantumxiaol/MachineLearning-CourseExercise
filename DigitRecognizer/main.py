
import torch
import torch.nn as nn
import pandas as pd

class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        item = self.data.iloc[index]
        label,features = item[0],torch.Tensor(item[1:])
        return features,label

def main():
    input_size = 784 #28*28
    num_classes = 10
    num_epochs = 5
    batch_size = 30
    learning_rate = 0.001
    data_train = pd.read_csv('./Data/train.csv')
    data_test = pd.read_csv('./Data/test.csv')
    train_dataset = DatasetMNIST(data_train)
    test_dataset = DatasetMNIST(data_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = nn.Linear(input_size,num_classes)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i ,(features,labels) in enumerate(train_loader):
            #前向传播
            outputs =model(features)
            loss = criterion(outputs,labels)

            #反向传播及优化
            #清空梯度缓存
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if(i+1) % 10 == 0 :
                print("Rpoch[{}/{}],Step[{}/{}],Loss:{:.4f}".format(epoch+1 , num_epochs , i+1 , total_step , loss.item()))

        with torch.no_grad():
            correct = 0
            total = 0
            for (features , labels) in test_loader:

                outputs = model(features)

                _ , predicted = torch.max(outputs.data , 1)


                total += labels.size(0)

                correct += (predicted == labels).sum()
            print("Accuracy of model on the 10000 test images:{}%".format(100*correct/total))

        total.save(model.state_dict() , "model.ckpt")

if __name__ =='__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
