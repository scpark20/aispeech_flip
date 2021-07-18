import json
import os
import torch

def sizeof_fmt(num, suffix='B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

class Logger:
    def __init__(self, save_dir, new=False):
        self.save_dir = save_dir
        self.json_file = save_dir + '/data.json'
        if os.path.exists(self.json_file) and not new:
            with open(self.json_file) as f:
                self.data = json.load(f)
        else:
            self.data = {'loss': [],
                         'test_loss': []}
            
    def log(self, step, key, value):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            
        self.data[key].append([step, value])
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f)
            
    def _get_save_path(self, save_dir, step):
        return save_dir + '/' + 'save_' + str(step)
    
    def save(self, step, model, optimizer):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            
        path = self._get_save_path(self.save_dir, step)
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)
        print('saved', path)
        
    def load(self, step, model, optimizer, path=None):
        if path is None:
            path = self._get_save_path(self.save_dir, step)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=torch.device('cpu'))    
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
            step = checkpoint['step']
            
            print('loaded :', step)
        
        return model, optimizer, step