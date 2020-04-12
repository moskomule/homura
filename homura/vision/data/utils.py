class _Segmentation(object):
    @staticmethod
    def cat_list(images, fill_value=0):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = _Segmentation.cat_list(images, fill_value=0)
        batched_targets = _Segmentation.cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


class _DetectionUtils(object):
    pass
