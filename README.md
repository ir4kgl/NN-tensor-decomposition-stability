# Повышение устойчивости обучения факторизованных слоев нейросетей с помощью функции чувствительности
*Исследовательский проект студента ФКН НИУ ВШЭ (2021/2022).*

Тензорные разложения могут использоваться в качестве инструмента сжатия нейронных сетей. Недостаток данного подхода — неустойчивость сжатой модели. С целью повышения устойчивости можно применить к факторизованным слоям нейронной сети алгоритм минимизации функции чувствительности тензорных разложений. Цель данной работы — обобщить определение функции чувствительности и реализовать алгоритм ее минимизации применительно к различным тензорным сетям.

### 2D-свертка
---

Набор обучаемых весов сверточного слоя класса `torch.nn.Conv2d` можно рассматривать как 3-мерный тензор размеров $$S \times T \times D^2$$, где $$S$$ -- число входных каналов, $$T$$ --  число выходных каналов, $$D$$ -- размер фильтра.

Данный тензор можно представить в виде 3 разложений: CPD (Canonical Polyadic decomposition), TKD (Tucker decomposition) и TC (tensor chain).

В модуле `tens_conv_2d.py` содержатся имплементации свёрточных слоев (2D-сверток) в виде тензорных разложений, а также метод `replace_conv_layer_2D`, заменяющий исходный сверточный слой модели на данное тензорное разложение.

##### Пример

```python
from tens_conv_2d import replace_conv_layer_2D, CPD_Conv_2D

model = torchvision.models.resnet18()
tn = CPD_Conv_2D
tn_args = {'rank': 10}
device = torch.device("cpu")

replace_conv_layer_2D(model, 'layer4', 1, 'conv1', tn, tn_args, device)
``` 


В модуле `tens_regularizer_2d.py` содержатся имплементации регуляризаторов тензорных разложений. Для того, чтобы применить регуляризацию во время обучения, достаточно к основной функции потерь (напр., `nn.CrossEntropyLoss`) прибавить штраф, найденный регуляризатором.

##### Пример

```python
from tens_regularizer_2d import CPD_Sensitivity_Regularizer_2D

criterion = torch.nn.CrossEntropyLoss()
regularizer = CPD_Sensitivity_Regularizer_2D('layer4', 1, 'conv1', regcoef=1e-3)
...

for (images, classes) in dataloader:
    loss = criterion(model(images), classes)
    loss += regularizer.calc_penalty(model)     # add penalty based on sensitivity function
    loss.backward()
    ...
``` 


