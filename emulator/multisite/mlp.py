"""
Four-head, multi-layer perceptron for emulator analysis.
"""
from torch import nn
import json
import os


class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(
        self,
        l1=32,
        l2=64,
        l3=128,
        l4=256,
        l5=512,
        l6=1024,
        l7=512,
        l8=256,
        l9=128,
        l10=64,
        l11=32,
        dropout_prob=0,
    ):
        super().__init__()
        # Generate the heads from the dimensions of each output
        with open(
            os.path.expanduser(
                "~/EMOD-calibration/emulator/output_dimensions_aggregate.json"
            ),
            "r",
        ) as content:
            dims = json.load(content)

        outputs = dims.keys()
        num_inputs = 8
        self.hidden_layers = nn.Sequential(
            nn.BatchNorm1d(num_inputs),
            nn.Linear(num_inputs, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, l4),
            nn.ReLU(),
            nn.Linear(l4, l5),
            nn.ReLU(),
            nn.Linear(l5, l6),
            nn.ReLU(),
        )

        head_layers = {}
        for output in outputs:
            dim_info = dims[output]
            out_dim = dim_info["end_idx"] - dim_info["begin_idx"]
            layers = [
                nn.Linear(l6, l7),
                nn.ReLU(),
                nn.Linear(l7, l8),
                nn.ReLU(),
                nn.Linear(l8, l9),
                nn.ReLU(),
                nn.Linear(l9, l10),
                nn.ReLU(),
                nn.Linear(l10, l11),
                nn.ReLU(),
                nn.Linear(l11, out_dim),
                nn.Dropout(p=dropout_prob),
            ]
            head_layers[output] = nn.Sequential(*layers)

        # Add individual head layers
        self.pfpr_layers = head_layers["PfPr"]
        self.incidence_layers = head_layers["Incidence"]
        self.gam1_layers = head_layers["Gametocytemia_1"]
        self.par1_layers = head_layers["Parasitemia_1"]
        self.par2_layers = head_layers["Parasitemia_2"]

    def forward(self, x):
        """
        Forward pass. Computes the different outputs individually.
        """
        outputs = {}
        sigmoid = nn.Sigmoid()
        x = self.hidden_layers(x)
        outputs["PfPr"] = sigmoid(self.pfpr_layers(x))
        outputs["Incidence"] = sigmoid(self.incidence_layers(x))
        outputs["Gametocytemia_1"] = sigmoid(self.gam1_layers(x))
        outputs["Parasitemia_1"] = sigmoid(self.par1_layers(x))
        outputs["Parasitemia_2"] = sigmoid(self.par2_layers(x))
        return outputs
