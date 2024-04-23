import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, MaxPool2d

from utils import SE3ToRt


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode="nearest", scale_factor=self.upsample
            )
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)


class BatchNormDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormDeconv2d, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.deconv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(FCN, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        # self.conv2d = Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.deconv2d(x)
        # x = self.conv2d(x)
        return F.relu(x, inplace=True)


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class PoseAndMaskEncoder(nn.Module):
    """Pose Nets"""

    def __init__(self, k, action_dim):
        # Implementation a mix of SE3 Nets and SE3 Pose Nets
        super(PoseAndMaskEncoder, self).__init__()
        self.k = k  # number of objects in scene
        self.action_dim = action_dim
        # Encoder
        self.Conv1 = Conv2d(3, 8, bias=False, kernel_size=2, stride=1, padding=1)
        self.Pool1 = MaxPool2d(kernel_size=2)
        self.Conv2 = BatchNormConv2d(
            8, 16, bias=False, kernel_size=3, stride=1, padding=1
        )
        self.Pool2 = MaxPool2d(kernel_size=2)
        self.Conv3 = BatchNormConv2d(
            16, 32, bias=False, kernel_size=3, stride=1, padding=1
        )
        self.Pool3 = MaxPool2d(kernel_size=2)
        self.Conv4 = BatchNormConv2d(
            32, 64, bias=False, kernel_size=3, stride=1, padding=1
        )
        self.Pool4 = MaxPool2d(kernel_size=2)
        self.Conv5 = BatchNormConv2d(
            64, 128, bias=False, kernel_size=3, stride=1, padding=1
        )
        self.Pool5 = MaxPool2d(kernel_size=2)
        # Mask Decoder
        self.Deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.Deconv2 = BatchNormDeconv2d(64, 32, kernel_size=3, stride=1, upsample=2)
        self.Deconv3 = BatchNormDeconv2d(32, 16, kernel_size=3, stride=1, upsample=2)
        self.Deconv4 = BatchNormDeconv2d(16, 8, kernel_size=3, stride=1, upsample=2)
        self.Deconv5 = FCN(8, self.k, kernel_size=3, stride=1, upsample=2)

        # SE3 Decoder
        self.Fc1 = Dense(128 * 7 * 7 + 256, 256, nn.ReLU())
        self.Fc2 = Dense(256, 128, nn.ReLU())
        self.Fc3 = Dense(128, 6 * self.k)

        # Action Encoder
        self.FC_action1 = Dense(self.action_dim, 128, nn.ReLU())
        self.FC_action2 = Dense(128, 256, nn.ReLU())

        # Latent to Image space
        self.FC_latent = Dense(128 * 7 * 7 + 256, 128 * 7 * 7, nn.ReLU())

    def encode_state(self, x):
        # x: 3 x 224 ** 2
        self.z1 = self.Pool1(self.Conv1(x))  # 8 x 112 ** 2
        self.z2 = self.Pool2(self.Conv2(self.z1))  # 16 x 56 ** 2
        self.z3 = self.Pool3(self.Conv3(self.z2))  # 32 x 28 ** 2
        self.z4 = self.Pool4(self.Conv4(self.z3))  # 64 x 14 ** 2
        self.z5 = self.Pool5(self.Conv5(self.z4))  # 128 x 7 ** 2
        z = self.z5
        return z

    def decode_mask(self, z):
        # z: 128 x 7 ** 2
        z = self.FC_latent(z).view(-1, 128, 7, 7)
        self.m1 = self.Deconv1(z)  # 64 x 14 ** 2
        self.m2 = self.Deconv2(self.m1 + self.z4)  # 32 x 28 ** 2
        self.m3 = self.Deconv3(self.m2 + self.z3)  # 16 x 56 ** 2
        self.m4 = self.Deconv4(self.m3 + self.z2)  # 8 x 112 ** 2
        self.m5 = self.Deconv5(self.m4 + self.z1)  # k x 224 ** 2
        # k_sum = torch.sum(self.m5, dim=1, keepdim=True)
        # m = self.m5 / k_sum  # sum of all mask weights per pixel equals 1
        m = F.softmax(self.m5,dim = 1)
        return m

    def decode_transform(self, latent):
        self.s1 = self.Fc1(latent)
        self.s2 = self.Fc2(self.s1)
        self.s3 = self.Fc3(self.s2)
        p = self.s3  # k x 6
        return p

    def upsample_action(self, action):
        x = self.FC_action1(action)
        x = self.FC_action2(x)  # 256
        return x

    def forward(self, x, action):
        enc_state = self.encode_state(x)
        action_upsampled = self.upsample_action(action)
        latent = torch.cat((enc_state.view(-1, 128 * 7 * 7), action_upsampled), axis=1)
        mask = self.decode_mask(latent)
        transforms = self.decode_transform(latent)
        return mask, transforms


class TransformNetwork(nn.Module):
    def __init__(self, k, action_dim, gamma=1.0 * 100, sigma=0.5 * 100, training=True):
        super(TransformNetwork, self).__init__()
        self.k = k
        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma
        self.training = training
        self.transform_creator = SE3ToRt()

    def sharpen_mask_weights(self, mask):
        """Apply weight sharpening

        Parameters
        ----------
        mask : array K , H , W
            softmax over object association for each pixel

        Returns
        -------
        array K , H * W
            sharpened weights
        """
        W = mask.shape[-1]
        H = mask.shape[-2]
        mask = mask.view(-1, self.k, H * W)
        eps = torch.normal(
            mean=torch.zeros(self.k, H * W),
            std=torch.ones(self.k, H * W) * self.sigma**2
        ).to(mask.device)
        mask = (mask + eps.unsqueeze(0)) ** self.gamma
        k_sum = torch.sum(mask, dim=1, keepdim=True)
        mask = mask / k_sum  # sum of all mask weights per pixel equals 1
        mask = mask.view(-1, self.k, H, W)
        if not self.training:
            ind = torch.max(mask, 1)[1]
            mask[torch.arange(len(ind)), ind] = 1.0

        return mask

    def transform_point_cloud(self, x, mask, transforms):
        """Transform the given point cloud via applying SE3 and masking

        Parameters
        ----------
        x : array 3 , H , W
            point cloud of scene, flattened
        mask : array K , H , W
            object mask (soft)
        transforms : K , 6
            translation/rotation

        Returns
        -------
        array 3 , H , W
            transformed point cloud of scene, unflattened
        """
        bs = x.shape[0]
        H = x.shape[-2]
        W = x.shape[-1]

        transforms = transforms.view(-1, self.k, 6)
        x = x.view(bs, 3, -1)  # flatten image (H*W)
        # get axang of current poses (poses is of dimension  6 * k, where k is the number of obj in scene)
        out_transforms = self.transform_creator(transforms)
        # R = axisAngleToRotationMatrix_batched(v)
        R = out_transforms[:, :, :, :3]
        T = out_transforms[:, :, :, 3]
        m = self.sharpen_mask_weights(mask)
        m = mask.view(bs, self.k, -1)
        m = m.unsqueeze(-1).expand(bs, self.k, H * W, 3).transpose(3, 2)
        # dim: K * 3 * (H*W)
        soft_transformed = torch.mul(
            m,
            torch.matmul(R, x.unsqueeze(1).expand(bs, self.k, 3, H * W))
            + T.unsqueeze(-1).expand(bs, self.k, 3, x.shape[-1]),
        )

        # dim: 3 * (H*W)
        x_new = torch.sum(soft_transformed, dim=1)
        x_new = x_new.view(bs, 3, H, W)
        return x_new

    def forward(self, x, mask, transforms):
        x_new = self.transform_point_cloud(x, mask, transforms)
        return x_new


class SE3Net(nn.Module):
    def __init__(self, k, action_dim, gamma=1.0, sigma=0.5, training=True):
        super(SE3Net, self).__init__()
        self.k = k
        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma
        self.training = training

        self.transformer = TransformNetwork(k, action_dim)
        self.encoder = PoseAndMaskEncoder(k, action_dim)

    def forward(self, x, action):
        self.mask, self.transforms = self.encoder(x, action)
        x_new = self.transformer(x, self.mask, self.transforms)
        return x_new


