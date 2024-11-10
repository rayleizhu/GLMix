from omegaconf import OmegaConf

f1 = 'configs/main_cx2.yaml'
f2 = 'configs/train_args.yaml'

conf1 = OmegaConf.load(f1)
dict1 = OmegaConf.to_container(conf1)

conf2 = OmegaConf.load(f2)
dict2 = OmegaConf.to_container(conf2)

for k,v in dict1.items():
    if k in dict2.keys() and dict2[k]!=dict1[k]:
        print(k, dict1[k], dict2[k])
