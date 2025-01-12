# [ICPR 2024] Re-Identification Based on the Spatial-Temporal Fusion Network



### 1. FusionNet 학습

---

1. fusionnet/scenario/auto.py - 설정 

   ```python
   # Root path
   root_path = '/home/cvlab/fast-reid/ICPR_code/'
   
   scenario_path = f'{root_path}/fusionnet/scenario'
   scenario = []
   
   # Dataset 선택 (veri, market)
   # data = 'veri'
   # data = 'market'
   
   splits = [5]
   fold_num = 5
   
   # FusionNet - Window 크기 선택 (W=0 ~ W=12)
   # scenario.append('scenario_w00')
   # scenario.append('scenario_w02')
   # scenario.append('scenario_w04')
   # scenario.append('scenario_w06')
   # scenario.append('scenario_w08')
   scenario.append('scenario_w10')
   # scenario.append('scenario_w12')
   
   # adaptive Parzen window 선택 (a=alpha, b=beta)
   parzens = [
               'a06-b25'
              ]
   ```

   - Root path
     - ICPR_code 폴더가 있는 경로
   - Dataset 선택 - VeRi776(=veri), Market-1501(=market)
   - FusionNet window 크기 선택
     - W=0 ~ 12
   - adaptive Parzen window 선택 
     - adaptive Parzen window를 수행한 파일들은 모두 아래 경로에 저장 (확인하기)
       - fusionnet/parzen_window_csv
     - a = alpha (scale factor)
     - b = beta (smoothness factor)
     - a06-b25
       - a = 6, b=25

2. 1번이 끝난 후 FusionNet 학습

   ```shell
   python auto.py
   ```

   

### 2. Appearance + FusionNet Inference

---

1. fastreid/auto.py - 설정 

```python
# Root path
root_path = '/home/cvlab/fast-reid/ICPR_code/'

scenario_path = f'{root_path}/fusionnet/scenario'
scenario = []

# Dataset 선택
data = 'veri'
# data = 'market'

# Appearance only: '0'
# Appearance + fusionnet: '5'
splits = [5]

fold_num = 5

# FusionNet - Window 크기 선택 
# scenario.append('scenario_w00')
# scenario.append('scenario_w02')
# scenario.append('scenario_w04')
# scenario.append('scenario_w06')
# scenario.append('scenario_w08')
scenario.append('scenario_w10')
# scenario.append('scenario_w12')


# adaptive Parzen window 선택 (a=alpha, b=beta)
parzens = [
           'a06-b25',
           ]
```

- 1번, FusionNet 학습과 동일하게 진행 
  - splits
    - 0번은 appearance model만 돌아감. FusionNet X
      - 모든 dataset을 그대로 학습시킨 모델 돌아감 
    - 5번은 appearance model + FusionNet
      - FusionNet 학습에 쓰이는 데이터를 제외하고 학습시킨 모델이 돌아감 

- 파라미터(dataset, split, FusionNet, adaptive Parzen window 등) 설정 후 inference code

  ```
  python auto.py
  ```

- Code Reference: FastReID
