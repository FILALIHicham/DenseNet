import torch
import torch.nn as nn
import torch.nn.init as init

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck, drop_rate):
        super(_DenseLayer, self).__init__()
        self.bottleneck = bottleneck
        self.drop_rate = drop_rate
        inter_channels = 4 * growth_rate if bottleneck else growth_rate

        layers = []
        if bottleneck:
            layers.extend([
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
            ])
            in_channels = inter_channels

        layers.extend([
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        ])

        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        new_features = self.layers(x)
        if self.dropout:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], dim=1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, drop_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate, growth_rate, bottleneck, drop_rate
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class _Transition(nn.Module):
    def __init__(self, in_channels, compression):
        super(_Transition, self).__init__()
        out_channels = int(in_channels * compression)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layers(x)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_layers=(16,16,16), 
                 num_init_features=24, bottleneck=True, compression=0.5,
                 drop_rate=0.2, num_classes=10):
        super(DenseNet, self).__init__()

        # Keep initial convolution separate
        self.init_conv = nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)
        channels = num_init_features

        self.features = nn.ModuleList()
        num_blocks = len(block_layers)
        for idx, num_layers in enumerate(block_layers):
            block = _DenseBlock(num_layers, channels, growth_rate, bottleneck, drop_rate)
            self.features.append(block)
            channels += num_layers * growth_rate

            if idx != num_blocks - 1:
                trans = _Transition(channels, compression)
                self.features.append(trans)
                channels = int(channels * compression)

        self.bn_final = nn.BatchNorm2d(channels)
        self.relu_final = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He (Kaiming) normal initialization for conv layers.
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Linear layers can also be initialized with Kaiming normal
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply the initial conv first
        out = self.init_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.relu_final(self.bn_final(out))
        out = torch.mean(out, dim=[2,3])  
        out = self.classifier(out)
        return out