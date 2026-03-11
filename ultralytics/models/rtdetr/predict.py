# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):

        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []  # Appearance-stream results.
        ir_results = []  # Motion-stream results.

        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)
            idx = max_score.squeeze(-1) > self.args.conf

            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx

            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]

            # Process the appearance image (last 3 channels).
            vis_img = orig_img[..., 3:] if orig_img.shape[-1] >= 6 else orig_img
            oh, ow = vis_img.shape[:2]
            pred_vis = pred.clone()
            pred_vis[..., [0, 2]] *= ow
            pred_vis[..., [1, 3]] *= oh
            results.append(Results(vis_img, path=img_path, names=self.model.names, boxes=pred_vis))

            # Process the motion image (first 3 channels).
            if orig_img.shape[-1] >= 6:
                ir_img = orig_img[..., :3]
                ir_oh, ir_ow = ir_img.shape[:2]
                pred_ir = pred.clone()
                pred_ir[..., [0, 2]] *= ir_ow
                pred_ir[..., [1, 3]] *= ir_oh

                # Build the paired motion-image path.
                ir_path = img_path.split('images')
                ir_path = str(ir_path[0] + 'image' + ir_path[1]) if len(ir_path) > 1 else img_path
                ir_results.append(Results(ir_img, path=ir_path, names=self.model.names, boxes=pred_ir))

        return results, ir_results

    def pre_transform(self, im):

        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]
