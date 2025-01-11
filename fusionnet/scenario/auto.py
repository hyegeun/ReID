import os

scenario = []

# Dataset 선택 
data = 'veri'
# data = 'market'

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
            'a06-b25'
           ]


def scenario_start(seeds, fold_num):
    for scene in scenario:
        try:
            print(scene)
            
            for split in splits:
                for parzen in parzens:
                    base = f'./{data}/{scene}/s{split}_{parzen}'
                    parzen_path = f'../parzen_window_csv/{data}/{parzen}.csv'

                    print(base)

                    cmd = (f'python {scene}.py --method {parzen_path} --base_path {base} --seed {seeds} --fold_num {fold_num} --data {data} --split {split}')
                    os.system(cmd)

        except Exception as e:
            print(f'{cmd} Error')


seed = 98764135 

scenario_start(seed, fold_num)