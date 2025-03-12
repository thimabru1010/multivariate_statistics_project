import torch
import torch.nn as nn
import torch.optim as optim

class DoubleConv(nn.Module):
    """Double convolution block used throughout UNet"""
    def __init__(self, in_channels, out_channels, activation='relu', dropout_rate=0.3):
        super(DoubleConv, self).__init__()
        
        if activation == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Activation {activation} not supported")
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_layer,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_layer,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)

class Unet_Autoencoder(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), depth=4, initial_filters=64, 
                 filter_growth_rate=2, dropout_rate=0.3, activation='relu'):
        """
        Flexible UNet Autoencoder with adjustable depth and filter counts
        
        Args:
            input_shape: Input image shape (channels, height, width) - PyTorch format
            depth: Number of downsampling/upsampling levels
            initial_filters: Number of filters in first layer
            filter_growth_rate: Multiplier for filters at each level
            dropout_rate: Dropout rate
            activation: Activation function to use
        """
        super(Unet_Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.depth = depth
        self.initial_filters = initial_filters
        self.filter_growth_rate = filter_growth_rate
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Create encoder, bottleneck, and decoder
        self._build_model()
        
    def _build_model(self):
        """Build the full UNet model architecture"""
        # Set up encoder (down) and decoder (up) paths
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Store filter sizes for each level to use in skip connections
        self.enc_filters = []
        
        # Encoder path
        in_channels = self.input_shape[0]
        for i in range(self.depth):
            out_channels = self.initial_filters * (self.filter_growth_rate ** i)
            self.down_path.append(
                DoubleConv(in_channels, out_channels, 
                           activation=self.activation, 
                           dropout_rate=self.dropout_rate)
            )
            self.enc_filters.append(out_channels)
            in_channels = out_channels
            
        # Bottleneck
        bottleneck_filters = in_channels * self.filter_growth_rate
        self.bottleneck = DoubleConv(
            in_channels, bottleneck_filters,
            activation=self.activation, 
            dropout_rate=self.dropout_rate
        )
        
        # Decoder path - CORREÇÃO: usar apenas (self.depth - 1) camadas de upsampling
        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        
        in_channels = bottleneck_filters
        # CORREÇÃO: Usar self.depth - 1 para igualar ao número de poolings
        for i in range(self.depth - 1):
            # Determine output channels for this decoder level
            level_idx = self.depth - i - 1
            out_channels = self.initial_filters * (self.filter_growth_rate ** level_idx)
            
            # Upsampling convolution
            self.up_convs.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            
            # Se temos skip connection, adicione esses canais à entrada
            skip_idx = self.depth - 2 - i
            skip_channels = self.enc_filters[skip_idx]
            in_channels_after_concat = out_channels + skip_channels
            
            # Double convolution after concatenation
            self.dec_convs.append(
                DoubleConv(in_channels_after_concat, out_channels,
                          activation=self.activation, 
                          dropout_rate=self.dropout_rate)
            )
            
            # Update in_channels for next iteration
            in_channels = out_channels
            
        # Output layer
        self.output_conv = nn.Conv2d(out_channels, self.input_shape[0], kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:  # Don't pool on last encoder layer
                skip_connections.append(x)
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path - CORREÇÃO: usar self.depth - 1 para percorrer o número correto de camadas
        for i in range(self.depth - 1):
            # Upsampling
            x = self.up_convs[i](x)
            
            # Skip connection
            skip_idx = self.depth - 2 - i
            skip_connection = skip_connections[skip_idx]
            
            # Ensure dimensions match
            if x.shape[2] != skip_connection.shape[2] or x.shape[3] != skip_connection.shape[3]:
                x = nn.functional.interpolate(
                    x, 
                    size=(skip_connection.shape[2], skip_connection.shape[3])
                )
                
            # Concatenate skip connection
            x = torch.cat([skip_connection, x], dim=1)
            
            # Apply double convolution
            x = self.dec_convs[i](x)
        
        # Output layer
        x = self.output_conv(x)
        return self.final_activation(x)
    
    def summary(self):
        """Print model summary"""
        channels, h, w = self.input_shape
        x = torch.zeros(1, channels, h, w)
        print(f"Input shape: {x.shape}")
        
        # Store intermediate features for visualization
        features = []
        skip_connections = []
        
        # Encoder
        for i, down in enumerate(self.down_path):
            x = down(x) 
            print(f"Encoder Block {i+1}: {x.shape}")
            features.append(x)
            
            if i < self.depth - 1:
                skip_connections.append(x)
                x = self.pool(x)
                print(f"After pooling: {x.shape}")
        
        # Bottleneck
        x = self.bottleneck(x)
        print(f"Bottleneck: {x.shape}")
        features.append(x)
        
        # Decoder - CORREÇÃO: usar self.depth - 1 no loop para corresponder ao número de upsamplings
        for i in range(self.depth - 1):
            # Upsampling
            x = self.up_convs[i](x)
            print(f"Decoder Upsampling {i+1}: {x.shape}")
            
            # Skip connection
            skip_idx = self.depth - 2 - i
            skip_connection = skip_connections[skip_idx]
            
            # Ensure dimensions match
            if x.shape[2] != skip_connection.shape[2] or x.shape[3] != skip_connection.shape[3]:
                x = nn.functional.interpolate(
                    x, 
                    size=(skip_connection.shape[2], skip_connection.shape[3])
                )
                
            # Concatenate skip connection
            x = torch.cat([skip_connection, x], dim=1)
            print(f"After skip connection: {x.shape}")
            
            # Apply double convolution
            x = self.dec_convs[i](x)
            print(f"Decoder Block {i+1}: {x.shape}")
            features.append(x)
        
        # Output layer
        x = self.output_conv(x)
        print(f"Output: {x.shape}")
        return features
    
    def get_latent_space_size(self):
        """Calculate and return the latent space dimensions"""
        _, h, w = self.input_shape
        for _ in range(self.depth - 1):
            h = h // 2
            w = w // 2
        filters = self.initial_filters * (self.filter_growth_rate ** self.depth)
        return (filters, h, w)
    
    def create_encoder(self):
        """Create an encoder-only model for feature extraction"""
        class Encoder(nn.Module):
            def __init__(self, unet):
                super(Encoder, self).__init__()
                self.down_path = unet.down_path
                self.pool = unet.pool
                self.bottleneck = unet.bottleneck
                self.depth = unet.depth
                
            def forward(self, x):
                for i, down in enumerate(self.down_path):
                    x = down(x)
                    if i < self.depth - 1:
                        x = self.pool(x)
                x = self.bottleneck(x)
                return x
        
        return Encoder(self)
