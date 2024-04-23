import torch
import torch.nn as nn
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(torch.autograd.Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    @staticmethod
    def forward(ctx, loc_data, conf_data, prior_data, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        ctx.num_classes = num_classes
        ctx.background_label = bkg_label
        ctx.top_k = top_k
        ctx.nms_thresh = nms_thresh
        ctx.conf_thresh = conf_thresh
        ctx.variance = cfg['variance']
        
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, ctx.num_classes, ctx.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    ctx.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, ctx.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, ctx.num_classes):
                c_mask = conf_scores[cl].gt(ctx.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, ctx.nms_thresh, ctx.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < ctx.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output