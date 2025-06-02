import torch
from torch import nn
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        batch_norm =True,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        # self.batch_norm = batch_norm
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # if scale_base == 0:
        #     self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # else:
        #     self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))    
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = batch_norm
            
    @torch.no_grad()
    def grid_rescale(self, x, updated_grid_size, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)
        self.grid_size = updated_grid_size
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid = grid.T.contiguous()
        self.register_buffer("grid", self.grid)
        # self.spline_weight.data = self.curve2coeff(x, unreduced_spline_output).contiguous()
        new_spline_weight = self.curve2coeff(x, unreduced_spline_output).contiguous()
        self.spline_weight = torch.nn.Parameter(new_spline_weight)

    def reset_parameters(self):
        if self.scale_base != 0:
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
            
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)


        if self.scale_base == 0:
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )
            output = spline_output
        
        elif self.scale_spline == 0:
            base_output = F.linear(self.base_activation(x), self.base_weight)
            output = base_output
        
        else:
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )
            base_output = F.linear(self.base_activation(x), self.base_weight)
            output = base_output + spline_output

        if self.batch_norm:
            output = self.batch_norm(output)

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        dropout_rate=0.5,  # 添加 dropout_rate 参数
        batch_norm = True
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.dropout_rate = dropout_rate
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    batch_norm = batch_norm
                )
            )
            self.layers.append(torch.nn.Dropout(p=self.dropout_rate))  # 在每个 KANLinear 层之后添加 Dropout 层

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if isinstance(layer, KANLinear):
                if update_grid:
                    layer.update_grid(x)
                x = layer(x)
            else:
                x = layer(x)  # Dropout 层
        return x

    def Grid_Rescale(self, x: torch.Tensor, updated_grid_size):
        for layer in self.layers:
            if isinstance(layer, KANLinear):
                layer.grid_rescale(x, updated_grid_size=updated_grid_size)
                x = layer(x)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers if isinstance(layer, KANLinear)
        )

class MLP(nn.Module):
    def __init__(self, layer_sizes, batch_norm =True, dropout_rate=0.0):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No dropout or activation on the last layer
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KAN_MLP(nn.Module):
    def __init__(self, kan_model, mlp_model):
        super(KAN_MLP, self).__init__()
        self.kan_model = kan_model
        self.mlp_model = mlp_model

    def forward(self, x):
        x = self.kan_model(x)
        mlp_output = self.mlp_model(x)
        return mlp_output

  
class KAN_MLP(nn.Module):
    def __init__(self,
                layers_hidden,
                KAN_first =True,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
                dropout_rate=0.5,  # 添加 dropout_rate 参数
                batch_norm = True):
        
        super(KAN_MLP, self).__init__()
        KAN_hiddens = layers_hidden[:-1]
        MLP_hiddens = layers_hidden[-2:]
        self.kan_model = KAN(
                            layers_hidden=KAN_hiddens,                
                            grid_size=grid_size,
                            spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            base_activation=torch.nn.SiLU,
                            grid_eps=grid_eps,
                            grid_range=[-1, 1],
                            dropout_rate=dropout_rate,  # 添加 dropout_rate 参数
                            batch_norm = batch_norm)
        self.mlp_model = MLP(layer_sizes = MLP_hiddens, 
                             batch_norm =batch_norm, 
                             dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.kan_model(x)
        x = self.mlp_model(x)
        return  x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_model.layers if isinstance(layer, KANLinear)
        )    

class MLP_KAN(nn.Module):
    def __init__(self, mlp_model,kan_model):
        super(MLP_KAN, self).__init__()
        self.kan_model = kan_model
        self.mlp_model = mlp_model

    def forward(self, x):
        x = self.mlp_model(x)
        kan_output = self.kan_model(x)
        return kan_output
       
class MLP_KAN(nn.Module):
    def __init__(self,
                layers_hidden,
                KAN_first =True,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
                dropout_rate=0.5,  # 添加 dropout_rate 参数
                batch_norm = True):
        
        super(MLP_KAN, self).__init__()
        MLP_hiddens = layers_hidden[:-1] 
        KAN_hiddens = layers_hidden[-2:] 
        self.mlp_model = MLP(layer_sizes = MLP_hiddens, 
                             batch_norm =batch_norm, 
                             dropout_rate=dropout_rate)
        self.kan_model = KAN(
                            layers_hidden=KAN_hiddens,                
                            grid_size=grid_size,
                            spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            base_activation=torch.nn.SiLU,
                            grid_eps=grid_eps,
                            grid_range=[-1, 1],
                            dropout_rate=dropout_rate,  # 添加 dropout_rate 参数
                            batch_norm = batch_norm)
    def forward(self, x):
        x = self.mlp_model(x)
        x = self.kan_model(x)
        return  x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_model.layers if isinstance(layer, KANLinear)
        ) 