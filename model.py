import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        x = self.embed(captions)
        x = torch.cat((features.unsqueeze(1), x), dim=1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted = []
        pred_len = 0
        while pred_len <= max_len:
            pred, states = self.rnn(inputs,states)
            pred = self.fc(pred.squeeze(dim = 1))
            _, i = torch.max(pred, 1)
            predicted.append(i.cpu().numpy()[0].item())
            if i == 1:
                break
            
            inputs = self.embed(i)
            inputs = inputs.unsqueeze(1)
            pred_len += 1
            
        return predicted

