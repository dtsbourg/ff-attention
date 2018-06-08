from ff_attention import FFAttention

class AdditionAttention(FFAttention):

    def embedding(self, x_t):
        # Initial = identity
        # h_t = f(x_t), f = id
        return x_t

    def activation(self, h_t):
        layer1 = torch.nn.Linear(self.n_features, self.out_dim)
        batch_norm = torch.nn.BatchNorm1d(self.n_features)
        return F.selu(self.layer1(h_t))

    def out(self, c):
        layer3 = torch.nn.Linear(self.n_features, self.hidden)
        c_= c.view(self.batch_size, self.out_dim, self.n_features)
        x = F.selu(layer3(c_))
        out_layer = torch.nn.Linear(self.hidden, self.out_dim)
        return F.tanh(out_layer(x))
