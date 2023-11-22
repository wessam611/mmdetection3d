# Copyright (c) OpenMMLab. All rights reserved.
from .cylinder3d_head import Cylinder3DHead
from .decode_head import Base3DDecodeHead
from .dgcnn_head import DGCNNHead
from .meta_kernel import MetaKernel
from .minkunet_head import MinkUNetHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .range_view_cnn_head import RangeViewCnnHead

__all__ = [
    'PointNet2Head', 'DGCNNHead', 'PAConvHead', 'Cylinder3DHead',
    'Base3DDecodeHead', 'MinkUNetHead', 'RangeViewCnnHead', 'MetaKernel'
]
