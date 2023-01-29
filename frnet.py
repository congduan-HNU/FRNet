class ours_v5_0(nn.Module):
    def __init__(self, **kwargs):
        # patch_size 为后面所接主干网络的w,h，
        super(ours_v5_0, self).__init__()
        inchannel = kwargs["inchannel"]
        num_classes = kwargs["num_classes"]
        image_size = kwargs["image_size"]
        init_channel = kwargs["init_channel"]
        patch_size = kwargs["patch_size"]
        try:
            backboneSize_H = kwargs["backboneSize_H"]
        except:
            backboneSize_H = 8
        if_eca = kwargs["if_eca"]
        self.eca_config = kwargs["eca_config"]

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        # num_patches 为切分后的总块数
        num_patches = (image_size // patch_size) ** 2
        # hidden_size 为中间的层数，一个通道信息输出深度应该为patch_size*patch_size
        # hidden_size = inchannel * patch_size ** 2 #v1版本
        hidden_size = 64  # v1版本

        self.patch = patch_size
        self.p = 1
        self.if_eca = if_eca
        self.h = backboneSize_H

        # 深度扩充部分
        self.expand = DepthPointConvModule(in_channel=inchannel, out_channel=init_channel, kernel_size=3, stride=1,
                                           padding=1)

        self.embedding = DepthPointConvModuleGroup(in_channel=init_channel, mid_channel=hidden_size, out_channel=hidden_size, stride=patch_size,
                                              kernel_size=patch_size, padding=0)

        # feature
        self.layer_0 = DepthPointConvModule(in_channel=self.p * num_patches, out_channel=256, stride=1)
        if self.if_eca and self.eca_config[0]:
            self.eca_layer_1 = eca_layer(256, 3)
        self.layer_1 = DepthPointConvModule(in_channel=256, out_channel=256, stride=2)
        self.layer_2 = DepthPointConvModule(in_channel=256, out_channel=512, stride=1)
        if self.if_eca and self.eca_config[1]:
            self.eca_layer_2 = eca_layer(512, 3)
        self.layer_3 = DepthPointConvModule(in_channel=512, out_channel=512, stride=2)
        self.layer_4 = DepthPointConvModule(in_channel=512, out_channel=512, stride=1)
        self.layer_5 = DepthPointConvModule(in_channel=512, out_channel=512, stride=1)
        if self.if_eca and self.eca_config[2]:
            self.eca_layer_3 = eca_layer(512, 3)
        self.layer_6 = ConvModule(in_channel=512, out_channel=512, kernel_size=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_7 = nn.Conv2d(512, num_classes, 1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.dropout_020 = nn.Dropout(p=0.20)

    def forward(self, x):
        x = self.expand(x)
        x = self.embedding(x)
        # 按给出的模式重组向量
        x = rearrange(x, 'b (d p) h w  -> b d p h w', p=self.p)
        x = rearrange(x, 'b d p h w  -> b (p h w) d')
        x = rearrange(x, 'b c (h w)  -> b c h w', h=self.h)
        x = self.layer_0(x)
        if self.if_eca and self.eca_config[0]:
            x = self.eca_layer_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        if self.if_eca and self.eca_config[1]:
            x = self.eca_layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        if self.if_eca and self.eca_config[2]:
            x = self.eca_layer_3(x)
        x = self.layer_6(x)
        x = self.avg_pool(x)
        x = self.dropout_020(x)
        x = self.layer_7(x)
        x = self.softmax(x)
        x = torch.flatten(x, 1)
        return 
