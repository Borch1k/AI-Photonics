"""Pipeline contract: one prediction, configurable updates, PNG reconstructions."""
from itertools import product
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import pipeline
from photonics import FFNPINN, OpticalConfig, Simulator, load_model
from optics import decode_phase, encode_phase


class PipelineContract(unittest.TestCase):
    def test_invalid_refinement_steps_fail_before_loading(self):
        for steps in (-1, 1.5):
            with self.subTest(steps=steps), patch.object(pipeline,'load_model') as loader:
                with self.assertRaisesRegex(ValueError,'non-negative integer'):
                    pipeline.run('lens',['missing.png']*4,'unused',refine_steps=steps)
                loader.assert_not_called()

    def test_both_modes_order_updates_and_png_contract(self):
        torch.set_num_threads(2)
        for mode,steps in product(('lens', 'no_lens'),(None,0,7)):
            with self.subTest(mode=mode,steps=steps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                models = root/'models'; models.mkdir()
                config = OpticalConfig(mode=mode)
                torch.manual_seed(123)
                model = FFNPINN(size=128, hidden_width=8)
                torch.save({'kind':'pinn', 'architecture':'ffn', 'hidden_width':8,
                            'optics':asdict(config), 'state_dict':model.state_dict()},
                           models/f'pinn_{mode}.pt')
                inputs=[]
                for index in range(4):
                    pixels=np.zeros((128,128),dtype=np.uint8)
                    pixels[12+index*9:55+index*9,20:80]=180+index*20
                    path=root/f'input_{index}.png'
                    Image.fromarray(pixels).save(path); inputs.append(path)
                events=[]
                predictions=[]
                original_step=torch.optim.Adam.step
                def step(optimizer,*args,**kwargs):
                    events.append('update')
                    return original_step(optimizer,*args,**kwargs)
                def loader(*args,**kwargs):
                    loaded,conf,ck=load_model(*args,**kwargs)
                    loaded.register_forward_hook(lambda *args:events.append('prediction'))
                    loaded.register_forward_hook(lambda model,inputs,result:predictions.append(result.detach().clone()))
                    return loaded,conf,ck
                def simulator(conf):
                    self.assertEqual(events,['prediction'])
                    return Simulator(conf)
                with patch.object(pipeline,'load_model',side_effect=loader), \
                     patch.object(pipeline,'Simulator',side_effect=simulator), \
                     patch.object(torch.optim.Adam,'step',new=step), \
                     patch.object(pipeline,'optimize',wraps=pipeline.optimize) as refinement:
                    output=root/'output'
                    kwargs={} if steps is None else {'refine_steps':steps}
                    paths=pipeline.run(mode,inputs,output,model_dir=models,**kwargs)
                self.assertEqual(events.count('prediction'),1)
                expected_steps=100 if steps is None else steps
                self.assertEqual(events.count('update'),expected_steps)
                self.assertEqual(refinement.call_count,int(expected_steps>0))
                self.assertEqual(len(list(output.iterdir())),5)
                for path in paths:
                    data=path.read_bytes()
                    self.assertEqual(data[:8],b'\x89PNG\r\n\x1a\n')
                    self.assertEqual(data[24:26],bytes([8,0]))  # 8-bit grayscale IHDR
                    with Image.open(path) as image:
                        self.assertEqual(image.size,(128,128))
                        self.assertEqual(image.mode,'L')
                with Image.open(output/'phase_mask.png') as image:
                    code=torch.from_numpy(np.array(image,copy=True))
                if steps==0:
                    torch.testing.assert_close(code,encode_phase(predictions[0][0]),rtol=0,atol=0)
                targets=torch.from_numpy(np.stack([np.array(Image.open(p),dtype=np.float32)/255 for p in inputs]))
                with torch.no_grad():
                    intensity=Simulator(config)(decode_phase(code)[None])[0]
                    expected=torch.round((intensity*targets.mean((-2,-1),keepdim=True)).clamp(0,1)*255).to(torch.uint8).numpy()
                for index in range(4):
                    with Image.open(output/f'reconstruction_{index+1}.png') as image:
                        np.testing.assert_array_equal(np.array(image),expected[index])


if __name__=='__main__':
    unittest.main()
