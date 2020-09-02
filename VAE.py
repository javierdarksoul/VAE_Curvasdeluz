import torch


class VAE(torch.nn.Module):
    
    def __init__(self, conv_channel=[1,3,6,12],n_latent=2):
        super(type(self), self).__init__()
        self.enc1=torch.nn.Conv1d(conv_channel[0],conv_channel[1],4,stride=2,padding=1)
        self.enc2=torch.nn.Conv1d(conv_channel[1], conv_channel[2],4,stride=2,padding=3)
        self.enc3=torch.nn.Conv1d(conv_channel[2], conv_channel[3],4,stride=2,padding=1)
        self.enc4=torch.nn.Linear(12*6 +1 ,128)
        self.enc5=torch.nn.Linear(128,64)
        self.enc_mu=torch.nn.Linear(64,n_latent)
        self.enc_logstd=torch.nn.Linear(64,n_latent)
        ##decode
        self.dec1=torch.nn.Linear(n_latent,64)
        self.dec2=torch.nn.Linear(64,128)
        self.dec3=torch.nn.Linear(128,12*6 +1)
        self.dec4=torch.nn.ConvTranspose1d(conv_channel[1],conv_channel[0],4,stride=2,padding=1)
        self.dec5=torch.nn.ConvTranspose1d(conv_channel[2],conv_channel[1],4,stride=2,padding=3)
        self.dec6=torch.nn.ConvTranspose1d(conv_channel[3],conv_channel[2],4,stride=2,padding=1)
        self.activation=torch.nn.ReLU()
        # Completar
    
    def encode(self, x,period):
        x=x.unsqueeze(1)
        x=self.activation(self.enc1(x))
        x=self.activation(self.enc2(x))
        x=self.activation(self.enc3(x))
        x=x.view(-1,x.shape[1]*x.shape[2])
        x=torch.cat((x,period.unsqueeze(-1)),dim=1)
        x=self.activation(self.enc4(x))
        x=self.activation(self.enc5(x))
        return self.enc_mu(x), self.enc_logstd(x)
        
    def decode(self, z,k):
        z=self.activation(self.dec1(z))
        z=self.activation(self.dec2(z))
        z=self.activation(self.dec3(z))
        #print(z.shape)
        z=z[:,:,0:72]
        #print(z.shape)
        z=z.view(-1,12,6)
        z=self.activation(self.dec6(z))
        z=self.activation(self.dec5(z))
        z=self.dec4(z)
        z=z.view(-1,40)
        return z
    
    def sample(self,enc_mu, enc_logstd,k=1):
        std=(0.5*enc_logstd).exp()
        e=torch.randn(enc_logstd.shape[0],k, enc_logstd.shape[1], device=std.device, requires_grad=False)
        return e.mul(std.unsqueeze(1)).add(enc_mu.unsqueeze(1))

    
    def forward(self, x,k,period):
        enc_mu, enc_logstd = self.encode(x,period)
        z=self.sample(enc_mu,enc_logstd,k)
        return enc_mu,enc_logstd,self.decode(z,k)
    
    def supremo_elbo(self,x,weight,period,samples=1):
        enc_mu,enc_logstd,dec_mu= self.forward(x,samples,period)
        KLD=-0.5 * (2.0 + enc_logstd - enc_mu.pow(2) - enc_logstd.exp().pow(2)).sum(-1)
        #print(KLD.sum())
        rec=-0.5*(((x - dec_mu).pow(2)))
        rec=(rec/weight.pow(2)).sum(dim=-1)
        #print(rec.sum())
        ELBO=torch.sum(KLD - rec)
        
        return ELBO,-rec.sum()/samples,KLD.sum()
    