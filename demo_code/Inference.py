import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from xdecoder.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from tqdm import tqdm

import cv2
import os
import glob
import subprocess
from PIL import Image
import random
import argparse
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES


os.chdir(os.path.dirname(os.path.abspath(__file__)))

def default_loader(path):
    return Image.open(path).convert('RGB')

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

def inference_image(model, image, reftxt, tasks='Text'):
    image_ori = transform(image)
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    data = {"image": images, "height": height, "width": width}
    
    if len(tasks) == 0:
        tasks = ["Panoptic"]
    
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False


    text = None
    if 'Text' in tasks:
        model.model.task_switch['grounding'] = True
        data['text'] = [reftxt]
    batch_inputs = [data]
    model.model.metadata = metadata
    results = model.model.evaluate(batch_inputs)
    pano_seg = results[-1]['panoptic_seg'][0]
    pano_seg_info = results[-1]['panoptic_seg'][1]
    demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
    res = demo.get_image()
    results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    if 'Text' in tasks:
        pred_masks = results['pred_masks'][0] # [101, 128, 128]
        v_emb = results['pred_captions'][0]
        t_emb = extra['grounding_class']

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        
        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id,:,:]# [1, 128, 128]
        pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]
    

    # interpolate mask to ori size
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    texts = [all_classes[pred_class[0]]]

    for idx, mask in enumerate(pred_masks_pos):
        # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
        out_txt = texts[idx] if 'Text' not in tasks else reftxt
        demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0]%133], text=out_txt)
    res = demo.get_image()
    torch.cuda.empty_cache()
    mask = np.uint8((pred_masks_pos[0])*255)
    # return Image.fromarray(res), stroke_inimg, stroke_refimg
    return Image.fromarray(res), Image.fromarray(mask).convert('RGB') ,None


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('--path_image', type=str, default='/data/hongyan/PaddleSeg/Matting/results_hy_test/outpainted_images_0.7/yawning_and_sleepy/rgba',
                        help='path to the image file')

    parser.add_argument('--conf_files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file')

    parser.add_argument('--tasks', default="Text", help="type of the task")
    
    parser.add_argument('--output_dir', type=str, default="/data/hongyan/Segment-Everything-Everywhere-All-At-Once/mql_test", help='path to output dir')

    parser.add_argument('--reftxt', type=str, default="person")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    mask_with_originimage_dir = os.path.join(args.output_dir, "mask_with_originimage")
    os.makedirs(mask_with_originimage_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    '''
    build args
    '''
    opt = load_opt_from_config_files(args.conf_files)
    opt = init_distributed(opt)

    # META DATA
    cur_model = 'None'
    if 'focalt' in args.conf_files:
        pretrained_pth = os.path.join("seem_focalt_v2.pt")
        if not os.path.exists(pretrained_pth):
            os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
        cur_model = 'Focal-T'
    elif 'focal' in args.conf_files:
        pretrained_pth = os.path.join("seem_focall_v1.pt")
        if not os.path.exists(pretrained_pth):
            os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
        cur_model = 'Focal-L'

    '''
    build model
    '''
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

    ref_txt = args.reftxt
    tasks = args.tasks

    # traverse the images
    images = glob.glob(os.path.join(args.path_image, '*.png'))

    for image_name in tqdm(images):
        image = default_loader(image_name)

        res_img, res_mask, _ = inference_image(model, image, ref_txt, tasks)
        image_name = image_name.split('/')[-1]
        res_img.save(os.path.join(mask_with_originimage_dir,image_name))
        res_mask.save(os.path.join(mask_dir, image_name))



if __name__ == '__main__':
    args = parse_args()
    main(args)