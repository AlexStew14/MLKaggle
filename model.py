import torch
import torch.nn as nn
from torch import optim

class BinaryClassification(nn.Module):
    def __init__(self, input_dim, width, depth, device, embed_sizes):        
        super(BinaryClassification, self).__init__()
        self.embed_sizes = embed_sizes
        if embed_sizes is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(cat, size) for cat, size in embed_sizes]).to(device=device)
            input_dim += sum(e.embedding_dim for e in self.embeddings)
            self.embed_dropout = nn.Dropout(.3)

        self.factor = 1
        layers = []
        layers.append(self._input_block(input_dim, width))
        for i in range(depth-1):
            layers.append(self._internal_block(width))
            width = int(width*self.factor)

        layers.append(self._output_block(width))
        
        self.model = nn.ModuleList(layers).to(device)
        self.model.apply(self.init_weights)

    def init_weights(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight)
            torch.nn.init.ones_(model.bias)

    def _input_block(self, input_dim, width):
        return nn.ModuleList([            
            nn.Linear(input_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(width),
        ])

    def _output_block(self, width):
        return nn.ModuleList([
            nn.Linear(width, 1),
            nn.Sigmoid()
        ])
    

    def _internal_block(self, width):
        return nn.ModuleList([
            nn.Linear(width, int(width*self.factor)),            
            nn.LeakyReLU(),
            nn.BatchNorm1d(int(width*self.factor)),
        ])


    def forward(self, X, X_embed):
        if self.embed_sizes is not None:
            X_embed = [e(X_embed[:, i])for i,e in enumerate(self.embeddings)]
            X_embed = torch.cat(X_embed, 1)
            X_embed = self.embed_dropout(X_embed)
            X = torch.cat([X, X_embed], 1)

        out = X
        for layer in self.model[0]:
            out = layer(out)

        layer_out = out
        for block in self.model[1:-1]:
            for layer in block:
                layer_out = layer(layer_out)

        for layer in self.model[-1]:
            out = layer(out)

        return out


if __name__ == '__main__':
    input_dim = 3
    width = 256
    depth = 8
    batch_size = 5    

    model = BinaryClassification(input_dim, width, depth, 'cpu', None)
    print(model.model)
    sample = torch.ones((batch_size, input_dim))
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters())

    # for i in range(100):
    #     opt.zero_grad()
    #     pred_y = model(sample, None)
    #     loss = criterion(pred_y, torch.zeros((batch_size,1), device='cpu'))
    #     loss.backward()
    #     opt.step()
    #     print(pred_y)

        
        