import os
import shutil

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


def configs(data_):            
    if data_ == 'market':
        config_file = './configs/Market1501/mgn_R101-ibn.yml'

    elif data == 'veri':
        config_file = './configs/VeRi/sbs_R50-ibn.yml'
    
    print(config_file)
    return config_file


def inference():
    for scene in scenario:
        try:
            for split in splits:
                for parzen in parzens:
                    config_file = configs(data) # model config file 
                    print(f'Config file = {config_file}\n')
                    
                    fusion_model = f'{scenario_path}/{data}/{scene}/s{split}_{parzen}/best.pth' # fusion model path 
                    print(f'Fusion model = {fusion_model}\n')

                    config = config_file.split('/')[-1]
                    ap_model = f'{root_path}/fast-reid/models/{data}/{config[:-4]}_s{split}/model_best.pth' # appearance model path 
                        
                    print(f'Appearance model = {ap_model}\n')

                    parzen_path = f'{root_path}/fusionnet/parzen_window_csv/{data}/{parzen}.csv'

                    output_dir = f'./logs/{data}/{scene}/s{split}_{parzen}'

                    if os.path.isdir(output_dir):
                        shutil.rmtree(output_dir)
                        os.makedirs(output_dir)
                    else:
                        os.makedirs(output_dir)

                    if split == 0:
                        cmd = f"python3 tools/train_net.py --config-file {config_file} --eval-only MODEL.WEIGHTS {ap_model} MODEL.DEVICE 'cuda:0' OUTPUT_DIR {output_dir}"
                    else:
                        cmd = f"python3 tools/train_net.py --config-file {config_file} --eval-only MODEL.WEIGHTS {ap_model} MODEL.DEVICE 'cuda:0' OUTPUT_DIR {output_dir} FUSIONNET {fusion_model} PARZEN {parzen_path}"
                        print(f'Parzen window = {parzen_path}\n')

                    
                    print(f'Output dir = {output_dir}\n')
                    print(cmd)
                    print('------------------------------------------------------------------')

                    os.system(cmd)
                        
        except Exception as e:
            print(f'{e} Error')

inference()