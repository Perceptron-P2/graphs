import pandas as pd
import matplotlib.pyplot as plt



#Reading the training dataset into a dataframe
train_data = pd.read_csv('iris_train.data')
#Adding column names to the dataframe
train_data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class_label']
#Reading the test dataset into a dataframe
test_data = pd.read_csv('iris_test.data')

frames = [train_data,test_data]
data = pd.concat(frames)

class0 = data[(data.class_label=='Iris-virginica')]
class1 = data[(data.class_label=='Iris-versicolor')]


fig = plt.figure()
fig.suptitle('Sepal Length vs Sepal Width',fontsize = 18)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.scatter(class0['sepal_length'], class0['sepal_width'], c='g', marker="s", label='Virginica')
plt.scatter(class1['sepal_length'], class1['sepal_width'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('plot1.png')
plt.show()



fig = plt.figure()
fig.suptitle('Sepal Length vs Petal Length',fontsize = 18)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Petal Length', fontsize=12)
plt.scatter(class0['sepal_length'], class0['petal_length'], c='g', marker="s", label='Virginica')
plt.scatter(class1['sepal_length'], class1['petal_length'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('plot2.png')
plt.show()


fig = plt.figure()
fig.suptitle('Sepal Length vs Petal Width',fontsize = 18)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)
plt.scatter(class0['sepal_length'], class0['petal_width'], c='g', marker="s", label='Virginica')
plt.scatter(class1['sepal_length'], class1['petal_width'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('plot3.png')
plt.show()


fig = plt.figure()
fig.suptitle('Sepal Width vs Petal Length',fontsize = 18)
plt.xlabel('Sepal Width', fontsize=12)
plt.ylabel('Petal Length', fontsize=12)
plt.scatter(class0['sepal_width'], class0['petal_length'],c='g', marker="s", label='Virginica')
plt.scatter(class1['sepal_width'], class1['petal_length'],c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('plot4.png')
plt.show()

fig = plt.figure()
fig.suptitle('Sepal Width vs Petal Width',fontsize = 18)
plt.xlabel('Sepal Width', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)
plt.scatter(class0['sepal_width'], class0['petal_width'],c='g', marker="s", label='Virginica')
plt.scatter(class1['sepal_width'], class1['petal_width'],c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('plot5.png')
plt.show()


fig = plt.figure()
fig.suptitle('Petal Length vs Petal Width',fontsize = 18)
plt.xlabel('Petal Length', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)
plt.scatter(class0['petal_length'], class0['petal_width'], c='g', marker="s", label='Virginica')
plt.scatter(class1['petal_length'], class1['petal_width'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('plot6.png')
plt.show()