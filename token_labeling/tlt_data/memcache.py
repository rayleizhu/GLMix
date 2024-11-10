import sys
sys.path.insert(0, '/mnt/lustre/share/pymc/py3') # hack
try:
    import mc
except ImportError as E:
    pass

from PIL import Image
import torch
import io

# Memcached: https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/TnJXc1Uj/page/2fbDMbpf
# TCS x Lustre: https://ones.ainewera.com/wiki/?from_wecom=1#/team/JNwe8qUX/space/TnJXc1Uj/page/H6FLCuD2
# .tar Package as Image Folder Dataset: https://gist.github.com/rwightman/5a7c9232eb57da20afa80cedb8ac2d38

def load_pil_img(raw_binary):
    img_str = mc.ConvertBuffer(raw_binary)
    buff = io.BytesIO(img_str)
    return Image.open(buff)

# https://stackoverflow.com/a/72045291
def load_torch_tensor(raw_binary):
    tensor_bytes = mc.ConvertBuffer(raw_binary)
    buff = io.BytesIO(tensor_bytes)
    return torch.load(buff)

class MemCachedLoader(object):

    def __init__(self, mclient_path, tcs:bool=True):
        
        assert mclient_path is not None, \
            "Please specify 'data_mclient_path' in the config."
        self.mclient_path = mclient_path
        server_list_config_file = "{}/server_list.conf".format(
            self.mclient_path) if not tcs else ""
        client_config_file = "{}/client.conf".format(self.mclient_path)

        self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                      client_config_file)
                                                      
    def __call__(self, file_path:str, load_type:str):
        assert load_type in {'pil_img', 'torch_tensor'}
        try:
            buffer = mc.pyvector() # prepare mem buffer
            self.mclient.Get(file_path, buffer)  # read raw_binary to buffer
            if load_type == 'pil_img':
                obj = load_pil_img(buffer)
            elif load_type == 'torch_tensor':
                obj = load_torch_tensor(buffer)
            else:
                raise NotImplementedError('Unkonwn load_type {}'.format(load_type))
        except:
            print('Read file failed ({})'.format(file_path))
            return None
        else:
            return obj