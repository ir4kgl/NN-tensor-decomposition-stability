# Повышение устойчивости обучения факторизованных слоев нейросетей с помощью функции чувствительности
*Исследовательский проект студентки ФКН НИУ ВШЭ (2021/2022).*

Тензорные разложения могут использоваться в качестве инструмента сжатия нейронных сетей. Недостаток данного подхода — неустойчивость сжатой модели. С целью повышения устойчивости можно применить к факторизованным слоям нейронной сети алгоритм минимизации функции чувствительности тензорных разложений. Цель данной работы — обобщить определение функции чувствительности и реализовать алгоритм ее минимизации применительно к различным тензорным сетям.

### 2D-свертка
---

Набор обучаемых весов сверточного слоя класса `torch.nn.Conv2d` можно рассматривать как 3-мерный тензор размеров (S, T, D^2), где S -- число входных каналов, T --  число выходных каналов, D -- размер фильтра.

Данный тензор можно представить в виде 3 разложений: CPD (Canonical Polyadic decomposition), TKD (Tucker decomposition) и TC (tensor chain).

В модуле `tens_conv_2d.py` содержатся имплементации свёрточных слоев (2D-сверток) в виде тензорных разложений, а также метод `replace_conv_layer_2D`, заменяющий исходный сверточный слой модели на данное тензорное разложение.

##### Пример

```python
from tens_conv_2d import replace_conv_layer_2D, CPD_Conv_2D

model = torchvision.models.resnet18()
tn = CPD_Conv_2D
tn_args = {'rank': 10}
device = torch.device("cpu")
submodule_name = 'layer1.1.conv2'

replace_conv_layer_2D(model, submodule_name, tn, tn_args, device)
``` 

### Регуляризатор функции чувствительности
---

Каждый имплементированный класс обладает методом  `calc_penalty()`, который вычисляет и возвращает значение функции чувствительности данного слоя.

Для того, чтобы применить регуляризацию чувствительности во время обучения, достаточно к основной функции потерь (напр., `nn.CrossEntropyLoss`) прибавить соответствующий штраф.

##### Пример

```python
from tens_regularizer_2d import CPD_Sensitivity_Regularizer_2D

criterion = torch.nn.CrossEntropyLoss()
reg_coef = 1e-6
replaced_layer = 'layer1.1.conv2'
...

for (images, classes) in dataloader:
    loss = criterion(model(images), classes)
    loss += reg_coef * model.get_submodule(replaced_layer).calc_penalty()    # add penalty based on sensitivity function
    loss.backward()
    ...
``` 

### Поиск устойчивого тензорного разложения
---

Альтернативный подход к повышению устойчивости сжатой модели -- поиск тензорного разложения сверточного слоя обученной модели с минимальным значением функции чувствительности.

Модуль `tens_correction.py` содержит методы поиска устойчивых тензорных разложений.

Для того, чтобы дообучить сжатую модель с найденным устойчивым тензорным разложением сверточного слоя, достаточно явно передать в метод `replace_conv_layer_2D` факторы тензорной сети.


##### Пример

```python
from tens_conv_2d import replace_conv_layer_2D, CPD_Conv_2D
from tens_correction import factorize_CPD

model = torchvision.models.resnet18(weights=ResNet18_Weights)
replaced_layer = 'layer1.1.conv2'
RANK = 10

K = model.get_submodule(replaced_layer).weight
args = {'delta' : 10, 'n_iters' : 5}
A, B, C = factorize_CPD(K, RANK, correct=True, correction_args=args)

tn = CPD_Conv_2D
tn_args = {'rank': RANK, 'factors' : (A, B, C)}
device = torch.device("cpu")

replace_conv_layer_2D(model, replaced_layer, tn, tn_args, device)

...
# fine-tuning the model
...
```





