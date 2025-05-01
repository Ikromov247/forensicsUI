from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models


class FeatureExtract:
    def __init__(self, model="resnet"):
        self.mps_device = torch.device("mps")
        if model == "resnet":
            self.model = self.load_resnet()
            self.preprocess = self.preprocess_resnet()
        elif model == "inception":
            self.model = self.load_inception()
            self.preprocess = self.preprocess_inception()

    @staticmethod
    def preprocess_resnet():
        return transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                   ])

    def preprocess_inception(self):
        return transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def load_resnet(self):
        resnet = models.resnet50()
        resnet.fc = torch.nn.Linear(in_features=2048, out_features=196)
        resnet.to(self.mps_device)
        loaded_state_dict = torch.load("models/ResNet50_v3.pth", map_location=self.mps_device)
        resnet.load_state_dict(loaded_state_dict)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        resnet.eval()
        return resnet

    def load_inception(self):
        model = models.inception_v3(pretrained=False, aux_logits=False)
        loaded_state_dict = torch.load("models/Inception.pth", map_location=self.mps_device)
        model.load_state_dict(loaded_state_dict, strict=False)
        model.fc = torch.nn.Identity()
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.mps_device)
        model.eval()
        return model

    def extract_vector(self, img_array):
        if isinstance(img_array, torch.Tensor):
            image = img_array
        else:
            image = Image.fromarray(img_array).convert('RGB')

        image = self.preprocess(image)
        image = torch.unsqueeze(image, 0).to(self.mps_device)  # Move image to MPS device
        with torch.no_grad():
            features = self.model(image)
        # features = torch.flatten(features, start_dim=1)
        return features.cpu().numpy()
